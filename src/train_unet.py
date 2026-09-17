import os
import sys

# Ensure Windows PATH includes conda environment's Library/bin for DLL resolution (matplotlib, numpy, etc.)
if sys.platform == "win32":
    env_dir = os.path.dirname(sys.executable)
    lib_bin = os.path.join(env_dir, "Library", "bin")
    if os.path.exists(lib_bin) and lib_bin not in os.environ["PATH"]:
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ["PATH"]

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom modules
from config import RAW_MTG_DIR, PROCESSED_RADAR_DIR, PROCESSED_AWOS_DIR, PATCH_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, NUM_WORKERS, IS_COLAB, H5_PATH
from dataloader import get_dataloaders, get_split_dataloaders
from model import MTGUNet, CloudUNet_Fusion
from loss import MaskedBCELoss
from metrics import compute_metrics, format_metrics

def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> tuple:
    """
    Calculates Accuracy, Precision, Recall, and F1-Score from predictions and targets,
    ignoring labels with value -1.
    """
    mask = targets != -1
    preds = preds[mask]
    targets = targets[mask]
    
    total = len(targets)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    tp = np.sum((preds == 1) & (targets == 1))
    tn = np.sum((preds == 0) & (targets == 0))
    fp = np.sum((preds == 1) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return accuracy, precision, recall, f1

from tqdm import tqdm

def buffered_shuffle_generator(dataloader, buffer_batches=128, batch_size=64):
    """
    Highly optimized generator that loads batches sequentially (fast HDF5 read),
    accumulates them in a buffer of 'buffer_batches', shuffles them, and yields batches.
    """
    buffer_features = []
    buffer_scalars = []
    buffer_labels = []
    
    for item in dataloader:
        if len(item) == 3:
            features, scalars, labels = item
        else:
            features, labels = item
            scalars = torch.zeros(labels.size(0), 3)
            
        buffer_features.append(features)
        buffer_scalars.append(scalars)
        buffer_labels.append(labels)
        
        if len(buffer_features) >= buffer_batches:
            flat_features = torch.cat(buffer_features, dim=0)
            flat_scalars = torch.cat(buffer_scalars, dim=0)
            flat_labels = torch.cat(buffer_labels, dim=0)
            
            num_samples = flat_features.size(0)
            indices = torch.randperm(num_samples)
            flat_features = flat_features[indices]
            flat_scalars = flat_scalars[indices]
            flat_labels = flat_labels[indices]
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                if end - start == batch_size:
                    yield flat_features[start:end], flat_scalars[start:end], flat_labels[start:end]
                    
            buffer_features.clear()
            buffer_scalars.clear()
            buffer_labels.clear()
            
    # Flush remaining
    if buffer_features:
        flat_features = torch.cat(buffer_features, dim=0)
        flat_scalars = torch.cat(buffer_scalars, dim=0)
        flat_labels = torch.cat(buffer_labels, dim=0)
        num_samples = flat_features.size(0)
        indices = torch.randperm(num_samples)
        flat_features = flat_features[indices]
        flat_scalars = flat_scalars[indices]
        flat_labels = flat_labels[indices]
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            yield flat_features[start:end], flat_scalars[start:end], flat_labels[start:end]

def train_one_epoch(model, dataloader, criterion, optimizer, device, limit_batches=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    is_h5 = False
    ds = dataloader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    from h5_dataset import MTGH5Dataset
    if isinstance(ds, MTGH5Dataset):
        is_h5 = True
        
    data_stream = dataloader
    if is_h5 and not getattr(ds, "pre_shuffled", False) and (limit_batches is None or limit_batches > 100):
        data_stream = buffered_shuffle_generator(dataloader, buffer_batches=32, batch_size=dataloader.batch_size)
        
    total_steps = limit_batches if limit_batches else len(dataloader)
    pbar = tqdm(enumerate(data_stream), total=total_steps, desc="  Training", leave=False, mininterval=5.0)
    for batch_idx, batch_item in pbar:
        if limit_batches and batch_idx >= limit_batches:
            break
            
        if len(batch_item) == 3:
            features, scalars, labels = batch_item
        else:
            features, labels = batch_item
            scalars = torch.zeros(labels.size(0), 3)
            
        features, scalars, labels = features.to(device), scalars.to(device), labels.to(device)
        
        # TRANSFORM FOR UNET
        # Create a spatial mask [Batch, 33, 33] initialized to -1 (ignore)
        b_size = features.size(0)
        spatial_labels = torch.full((b_size, 33, 33), -1, dtype=torch.long, device=device)
        # Assign the station label to the center pixel (16, 16)
        spatial_labels[:, 16, 16] = labels
        
        optimizer.zero_grad()
        
        # Dispatch inputs based on model expectation
        if hasattr(model, 'module'):
            uses_scalars = 'Fusion' in model.module.__class__.__name__
        else:
            uses_scalars = 'Fusion' in model.__class__.__name__
            
        if uses_scalars:
            outputs = model(features, scalars)
        else:
            outputs = model(features)
        
        # CrossEntropyLoss expects output [B, C, H, W] and target [B, H, W]
        loss = criterion(outputs, spatial_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Get predictions
        _, preds = torch.max(outputs, 1) # preds is [Batch, 33, 33]
        
        # Only evaluate at the center pixel where we have ground truth
        center_preds = preds[:, 16, 16]
        all_preds.extend(center_preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    num_batches = limit_batches if limit_batches and limit_batches < len(dataloader) else len(dataloader)
    epoch_loss = running_loss / num_batches
    acc, prec, rec, f1 = calculate_metrics(np.array(all_preds), np.array(all_targets))
    
    return epoch_loss, acc, f1

def validate(model, dataloader, criterion, device, limit_batches=None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    total_steps = limit_batches if limit_batches else len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=total_steps, desc="  Validating", leave=False, mininterval=5.0)
    with torch.no_grad():
        for batch_idx, batch_item in pbar:
            if limit_batches and batch_idx >= limit_batches:
                break
                
            if len(batch_item) == 3:
                features, scalars, labels = batch_item
            else:
                features, labels = batch_item
                scalars = torch.zeros(labels.size(0), 3)
                
            features, scalars, labels = features.to(device), scalars.to(device), labels.to(device)
            
            # TRANSFORM FOR UNET
            b_size = features.size(0)
            spatial_labels = torch.full((b_size, 33, 33), -1, dtype=torch.long, device=device)
            spatial_labels[:, 16, 16] = labels
            
            if hasattr(model, 'module'):
                uses_scalars = 'Fusion' in model.module.__class__.__name__
            else:
                uses_scalars = 'Fusion' in model.__class__.__name__
                
            if uses_scalars:
                outputs = model(features, scalars)
            else:
                outputs = model(features)
            loss = criterion(outputs, spatial_labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            center_preds = preds[:, 16, 16]
            all_preds.extend(center_preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs[:, :, 16, 16], 1)[:, 1].cpu().numpy())
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    num_batches = limit_batches if limit_batches and limit_batches < len(dataloader) else len(dataloader)
    val_loss = running_loss / max(num_batches, 1)
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets), probs=np.array(all_probs))
    
    return val_loss, metrics


def evaluate_partial(model, dataloader, device, limit_batches=None):
    """
    test_partial (1-4 okta, etiketsiz) tutarlılık testi: okta başına ortalama bulut olasılığı ve
    'bulutlu' tahmin oranı. Beklenti: olasılık okta ile monoton artmalı, 0 ve >=5 okta arasında kalmalı.
    """
    model.eval()
    subset = dataloader.dataset
    base = subset
    while hasattr(base, "dataset"):
        base = base.dataset
    if getattr(base, "okta", None) is None:
        return {}
    okta_all = np.asarray(base.okta)[np.asarray(subset.indices)]
    probs = []
    with torch.no_grad():
        for batch_idx, batch_item in enumerate(dataloader):
            if limit_batches and batch_idx >= limit_batches:
                break
            features, scalars, labels = batch_item if len(batch_item) == 3 else (batch_item[0], torch.zeros(batch_item[1].size(0), 3), batch_item[1])
            features, scalars = features.to(device), scalars.to(device)
            uses_scalars = 'Fusion' in (model.module if hasattr(model, 'module') else model).__class__.__name__
            outputs = model(features, scalars) if uses_scalars else model(features)
            probs.extend(torch.softmax(outputs[:, :, 16, 16], 1)[:, 1].cpu().numpy())
    probs = np.asarray(probs); okta_all = okta_all[:len(probs)]
    out = {}
    for o in sorted(set(okta_all.tolist())):
        m = okta_all == o
        out[f"okta{o}"] = {"n": int(m.sum()), "mean_prob": float(probs[m].mean()), "frac_cloudy": float((probs[m] >= 0.5).mean())}
    out["all"] = {"n": int(len(probs)), "mean_prob": float(probs.mean()), "frac_cloudy": float((probs >= 0.5).mean())}
    return out

def run_experiment(name: str, train_loader, val_loader, epochs: int = EPOCHS, limit_batches: int = None, test_loaders: dict = None):
    print(f"\n==========================================")
    print(f"Starting Experiment: {name}")
    print(f"Model Type: UNet")
    print(f"Epochs: {epochs} | Limit Batches: {limit_batches}")
    print(f"==========================================")
    
    _ds = train_loader.dataset
    while hasattr(_ds, "dataset"):
        _ds = _ds.dataset
    in_ch = int(getattr(_ds, "num_channels", 17))
    print(f"Input channels: {in_ch}")
    
    if name == "CloudUNet_Fusion":
        model = CloudUNet_Fusion(in_channels=in_ch, scalar_features=3, num_classes=2).to(DEVICE)
    else:
        model = MTGUNet(in_channels=in_ch, num_classes=2).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    ds = train_loader.dataset
    if hasattr(ds, 'dataset'): # Wrapped by AugmentedDataset
        subset = ds.dataset
        base_ds = subset.dataset
        indices = subset.indices
    else: # Standard Subset
        base_ds = ds.dataset
        indices = ds.indices
        
    if hasattr(base_ds, 'labels'):
        train_labels = base_ds.labels[indices]
    else:
        train_labels = []
        for idx in indices:
            cc = base_ds.patch_samples[idx]['cloud_coverage']
            if 1 <= cc <= 8:
                train_labels.append(1)
            elif cc == 0:
                train_labels.append(0)
            else:
                train_labels.append(-1)
        train_labels = np.array(train_labels)
        
    valid_train_labels = train_labels[train_labels != -1]
    class_counts = np.bincount(valid_train_labels, minlength=2)
    total_valid = len(valid_train_labels)
    
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(total_valid / (2.0 * count))
        else:
            class_weights.append(1.0)
            
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"  Training label counts: Class 0 (Clear) = {class_counts[0]}, Class 1 (Cloudy) = {class_counts[1]}")
    print(f"  Dynamic Class Weights: {class_weights}")
    
    criterion = MaskedBCELoss(ignore_index=-1, weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "val_prec": [], "val_rec": [], "val_bal_acc": [], "val_mcc": [], "val_f1_clear": [], "val_auc": []
    }
    
    best_f1 = 0.0   # model seçimi ölçütü: val BALANCED ACCURACY
    checkpoint_path = f"checkpoint_{name.lower()}.pth"
    start_epoch = 1
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint['best_f1']
            history = checkpoint['history']
            print(f"-> Resuming from epoch {start_epoch} using existing checkpoint.")
        except Exception as e:
            print(f"-> Warning: Could not load checkpoint {checkpoint_path} ({e}). Starting from epoch 1.")
            start_epoch = 1
            
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, limit_batches)
        val_loss, vm = validate(model, val_loader, criterion, DEVICE, limit_batches)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["train_f1"].append(float(train_f1))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(vm["accuracy"])
        history["val_prec"].append(vm["precision"])
        history["val_rec"].append(vm["recall"])
        history["val_f1"].append(vm["f1_cloudy"])
        history["val_bal_acc"].append(vm["balanced_accuracy"])
        history["val_mcc"].append(vm["mcc"])
        history["val_f1_clear"].append(vm["f1_clear"])
        history["val_auc"].append(vm["roc_auc"])
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | {format_metrics(vm)}")
        
        if vm["balanced_accuracy"] > best_f1:
            best_f1 = vm["balanced_accuracy"]
            torch.save(model.state_dict(), f"best_model_{name.lower()}.pth")
            print(f"  --> Saved new best model: best_model_{name.lower()}.pth")
            
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            pass
            
    elapsed = time.time() - start_time
    print(f"Finished {name} in {elapsed/60:.2f} minutes. Best Val BalancedAcc: {best_f1:.4f}")
    
    test_results = {}
    if test_loaders:
        best_path = f"best_model_{name.lower()}.pth"
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=False))
        for tname, tloader in test_loaders.items():
            if tname == "test_partial":
                pr = evaluate_partial(model, tloader, DEVICE, limit_batches)
                test_results[tname] = pr
                print("  [test_partial] " + ", ".join(f"{k}: p̄={v['mean_prob']:.2f} cloudy%={v['frac_cloudy']*100:.0f} (n={v['n']})" for k, v in pr.items()))
                continue
            _, tm = validate(model, tloader, criterion, DEVICE, limit_batches)
            test_results[tname] = tm
            print(f"  [{tname}] {format_metrics(tm)}")
    
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception as e:
            pass
            
    return history, best_f1, num_params, test_results

def main():
    import argparse, random
    parser = argparse.ArgumentParser(description="Train U-Net models")
    parser.add_argument("--model", type=str, default="unet", help="unet (MTGUNet, aux kanallı tam harita) | unet_fusion (CloudUNet_Fusion) | all")
    parser.add_argument("--split", type=str, default=None, help="make_splits.py çıktısı (.npz)")
    parser.add_argument("--aux", action="store_true", help="H5 'aux' kanallarını girdiye ekle")
    parser.add_argument("--aux_channels", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    aux_channels = args.aux_channels.split(",") if args.aux_channels else None
    
    all_experiments = {"MTGUNet": "MTGUNet", "CloudUNet_Fusion": "CloudUNet_Fusion"}
    if args.model == "unet":
        experiments = {"MTGUNet": "MTGUNet"}
    elif args.model == "unet_fusion":
        experiments = {"CloudUNet_Fusion": "CloudUNet_Fusion"}
    else:
        experiments = all_experiments
    
    print("Initializing Datasets and DataLoaders...")
    test_loaders = None
    if args.split:
        loaders = get_split_dataloaders(H5_PATH, args.split, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                        use_aux=args.aux, aux_channels=aux_channels, shuffle_train=False)
        train_loader, val_loader = loaders["train"], loaders["val"]
        test_loaders = {k: v for k, v in loaders.items() if k.startswith("test")}
    elif os.path.exists(H5_PATH):
        print(f"HDF5 dataset found at {H5_PATH}. Using fast H5 loading!")
        train_loader, val_loader = get_dataloaders(
            h5_path=H5_PATH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            use_aux=args.aux,
            aux_channels=aux_channels
        )
    else:
        print(f"HDF5 dataset NOT found at {H5_PATH}. Falling back to NetCDF/Numpy loading.")
        train_loader, val_loader = get_dataloaders(
            mtg_dir=RAW_MTG_DIR,
            radar_dir=PROCESSED_RADAR_DIR,
            awos_dir=PROCESSED_AWOS_DIR,
            mode="patch",
            patch_size=PATCH_SIZE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=IS_COLAB
        )
    
    import json
    os.makedirs("artifacts", exist_ok=True)
    json_path = args.results or os.path.join("artifacts", "unet_results.json")
    results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            pass
    
    # Run a dry-run limit if not on colab to verify it works quickly
    epochs_to_run = args.epochs if args.epochs is not None else (2 if not IS_COLAB else 15)
    limit_batches = args.limit_batches if args.limit_batches is not None else (20 if not IS_COLAB else None)
    
    for name, model_type in experiments.items():
        if name in results:
            print(f"Model '{name}' already trained. Skipping.")
            continue
            
        history, best_f1, num_params, test_results = run_experiment(
            name, 
            train_loader,
            val_loader,
            epochs=epochs_to_run, 
            limit_batches=limit_batches,
            test_loaders=test_loaders
        )
        results[name] = {
            "history": history,
            "best_f1": best_f1,
            "best_val_balanced_accuracy": best_f1,
            "num_params": num_params,
            "test": test_results,
            "config": {"seed": args.seed, "aux": args.aux, "aux_channels": aux_channels, "split": args.split,
                       "epochs": epochs_to_run, "limit_batches": limit_batches}
        }
        
        try:
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            pass
        
    print("\n" + "="*50)
    print("         U-NET RESULTS")
    print("="*50)
    print(f"{'Model Name':<18} | {'Params':<10} | {'Val BalAcc':<10} | {'Test_time BalAcc/MCC':<22} | {'Test_station BalAcc/MCC':<22}")
    print("-"*95)
    for name, res in results.items():
        t = res.get("test", {}) or {}
        f = lambda m: f"{m['balanced_accuracy']:.4f}/{m['mcc']:.4f}" if m else "-"
        print(f"{name:<18} | {res['num_params']:<10,} | {res['best_f1']:<10.4f} | {f(t.get('test_time')):<22} | {f(t.get('test_station')):<22}")
    
if __name__ == "__main__":
    main()
