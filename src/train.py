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
from model import MTGConvNet
from metrics import compute_metrics, format_metrics

# --- Çıktı dizini: checkpoint / en iyi model / sonuç JSON buraya yazılır.
# Colab'da Drive altındaki kalıcı bir yol verin (--outdir), böylece oturum çökse de kaybolmaz.
OUT_DIR = os.environ.get("MTGCLM_OUT", "artifacts")


def out_path(*parts) -> str:
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p


def atomic_save(obj, path: str):
    """Önce .tmp'ye yaz, sonra yerine taşı: yazma sırasında çökme dosyayı bozmaz."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path: str):
    import json as _json
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(obj, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_load(path: str, device):
    """Bozuk/yarım checkpoint'i sessizce yok sayar (çökme anında yazılmış olabilir)."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  --> Uyarı: {os.path.basename(path)} okunamadı ({e}); yok sayılıyor.")
        return None


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
    buffer_batches = 128 batches = 8,192 samples.
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
    
    # Auto-detect if using H5 dataset
    is_h5 = False
    ds = dataloader.dataset
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    from h5_dataset import MTGH5Dataset
    if isinstance(ds, MTGH5Dataset):
        is_h5 = True
        
    data_stream = dataloader
    # Wrap with buffered shuffle only if it's H5 and we are training (shuffling is needed)
    # limit_batches check is to prevent wrapping for tiny dry-runs
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
        
        # CrossEntropyLoss ignores -1 automatically if ignore_index is set
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
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
            
            if hasattr(model, 'module'):
                uses_scalars = 'Fusion' in model.module.__class__.__name__
            else:
                uses_scalars = 'Fusion' in model.__class__.__name__
                
            if uses_scalars:
                outputs = model(features, scalars)
            else:
                outputs = model(features)
                
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
            
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
            probs.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
    probs = np.asarray(probs); okta_all = okta_all[:len(probs)]
    out = {}
    for o in sorted(set(okta_all.tolist())):
        m = okta_all == o
        out[f"okta{o}"] = {"n": int(m.sum()), "mean_prob": float(probs[m].mean()), "frac_cloudy": float((probs[m] >= 0.5).mean())}
    out["all"] = {"n": int(len(probs)), "mean_prob": float(probs.mean()), "frac_cloudy": float((probs >= 0.5).mean())}
    return out

def run_experiment(name: str, conv_blocks: list, train_loader, val_loader, epochs: int = EPOCHS, limit_batches: int = None, use_gap: bool = True, model_type: str = "ConvNet", test_loaders: dict = None):
    """
    Runs training and validation for a specific CNN or ResNet architecture configuration.
    Supports epoch-level checkpointing for resuming interrupted runs.
    """
    print(f"\n==========================================")
    print(f"Starting Experiment: {name}")
    print(f"Model Type: {model_type} | Blocks: {conv_blocks} | Use GAP: {use_gap}")
    print(f"Epochs: {epochs} | Limit Batches: {limit_batches}")
    print(f"==========================================")
    
    # Kanal sayısı veri setinden (17 uydu/radar + aux kanalları)
    _ds = train_loader.dataset
    while hasattr(_ds, "dataset"):
        _ds = _ds.dataset
    in_ch = int(getattr(_ds, "num_channels", 17))
    print(f"Input channels: {in_ch}")
    
    # Initialize model
    if model_type == "ResNet18":
        from model import MTGResNet18
        model = MTGResNet18(in_channels=in_ch, num_classes=2).to(DEVICE)
    elif model_type == "ResNet18_LateFusion":
        from model import ResNet18_LateFusion
        model = ResNet18_LateFusion(in_channels=in_ch, scalar_features=3, num_classes=2).to(DEVICE)
    elif model_type == "MTGConvNet_LateFusion":
        from model import MTGConvNet_LateFusion
        model = MTGConvNet_LateFusion(in_channels=in_ch, scalar_features=3, num_classes=2, conv_blocks=conv_blocks, dropout_rate=0.2).to(DEVICE)
    else:
        from model import MTGConvNet
        model = MTGConvNet(in_channels=in_ch, num_classes=2, conv_blocks=conv_blocks, use_gap=use_gap).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Unpack to get base dataset and training indices for class weights
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
    
    # Set Loss with ignore_index=-1 and dynamic class weighting
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights_tensor)
    
    # AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine Annealing Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "val_prec": [], "val_rec": [], "val_bal_acc": [], "val_mcc": [], "val_f1_clear": [], "val_auc": []
    }
    
    best_f1 = 0.0   # model seçimi ölçütü: val BALANCED ACCURACY (adı geriye uyumluluk için korunuyor)
    checkpoint_path = out_path("checkpoints", f"checkpoint_{name.lower()}.pth")
    best_model_path = out_path("checkpoints", f"best_model_{name.lower()}.pth")
    start_epoch = 1
    
    # Try to load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = safe_load(checkpoint_path, DEVICE)
            if checkpoint is None:
                raise RuntimeError("bozuk checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint['best_f1']
            history = checkpoint['history']
            print(f"-> Resuming from epoch {start_epoch} using existing checkpoint (Best F1 so far: {best_f1:.4f}).")
        except Exception as e:
            print(f"-> Warning: Could not load checkpoint {checkpoint_path} ({e}). Starting from epoch 1.")
            start_epoch = 1
            
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, limit_batches)
        val_loss, vm = validate(model, val_loader, criterion, DEVICE, limit_batches)
        
        scheduler.step() # Step learning rate
        
        epoch_time = time.time() - epoch_start
        
        # Save history (numpy skalerleri float'a çevrilir: JSON/torch.load uyumu)
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
        
        # Save best model (balanced accuracy)
        if vm["balanced_accuracy"] > best_f1:
            best_f1 = vm["balanced_accuracy"]
            atomic_save(model.state_dict(), best_model_path)
            print(f"  --> Yeni en iyi model kaydedildi: {best_model_path}")
            
        # Save epoch checkpoint to allow resume
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'history': history
            }
            atomic_save(checkpoint, checkpoint_path)
        except Exception as e:
            print(f"  --> Warning: could not save checkpoint: {e}")
            
    elapsed = time.time() - start_time
    print(f"Finished {name} in {elapsed/60:.2f} minutes. Best Val BalancedAcc: {best_f1:.4f}")
    
    # --- Test kümeleri: en iyi (val) modelle, yalnızca raporlama için
    test_results = {}
    if test_loaders:
        if os.path.exists(best_model_path):
            sd = safe_load(best_model_path, DEVICE)
            if sd is not None:
                model.load_state_dict(sd)
        for tname, tloader in test_loaders.items():
            if tname == "test_partial":
                pr = evaluate_partial(model, tloader, DEVICE, limit_batches)
                test_results[tname] = pr
                print("  [test_partial] " + ", ".join(f"{k}: p̄={v['mean_prob']:.2f} cloudy%={v['frac_cloudy']*100:.0f} (n={v['n']})" for k, v in pr.items()))
                continue
            _, tm = validate(model, tloader, criterion, DEVICE, limit_batches)
            test_results[tname] = tm
            print(f"  [{tname}] {format_metrics(tm)}")
    
    # Remove checkpoint file upon successful completion of the experiment
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"-> Removed temporary checkpoint file {checkpoint_path}")
        except Exception as e:
            print(f"-> Warning: could not remove checkpoint file: {e}")
            
    return history, best_f1, num_params, test_results

import argparse

def main():
    parser = argparse.ArgumentParser(description="Train MTGCLM Models")
    parser.add_argument("--model", type=str, default="all",
                        help="all | arch (6 mimari varyantı) | arch_flat | shallow_flat | medium_flat | deep_flat | shallow_gap | medium_gap | deep_gap | resnet18 | mtg_late_fusion | resnet18_fusion | mtg_flat (=Deep_Flat)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader işçi sayısı (config.NUM_WORKERS yerine). Colab T4 için 4 önerilir; 0 = ana işlemde oku (yavaş).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Checkpoint / en iyi model / sonuç JSON dizini (Colab: Drive altında kalıcı bir yol). Varsayılan: MTGCLM_OUT ortam değişkeni ya da artifacts/")
    parser.add_argument("--h5", type=str, default=None, help="HDF5 yolu (config.H5_PATH / MTGCLM_H5 yerine geçer)")
    parser.add_argument("--split", type=str, default=None, help="make_splits.py çıktısı (.npz). Verilirse train/val/test_* buradan gelir.")
    parser.add_argument("--aux", action="store_true", help="H5 'aux' kanallarını (t2m, rh2m, elev, cos_sza, radar_cov) girdiye ekle")
    parser.add_argument("--aux_channels", type=str, default=None, help="Virgülle ayrılmış aux alt kümesi, ör. t2m,rh2m,cos_sza")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Varsayılan: Colab 15, yerel 3")
    parser.add_argument("--limit_batches", type=int, default=None, help="Varsayılan: Colab None, yerel 30")
    parser.add_argument("--results", type=str, default=None, help="Sonuç JSON yolu (varsayılan artifacts/ablation_results.json)")
    args = parser.parse_args()
    
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    aux_channels = args.aux_channels.split(",") if args.aux_channels else None
    global OUT_DIR
    if args.outdir:
        OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'checkpoints'), exist_ok=True)
    n_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    h5_path = args.h5 or H5_PATH
    print(f"HDF5:   {h5_path}")
    print(f"Çıktı:  {os.path.abspath(OUT_DIR)}")
    print(f"DataLoader işçisi: {n_workers}")
    
    # Configurations to compare (Ablation Study)
    # (conv_blocks, use_gap, model_type)
    SHALLOW, MEDIUM, DEEP = [32, 64], [32, 64, 128], [32, 64, 128, 256]
    all_experiments = {
        # --- mimari ablasyonu: derinlik (Shallow/Medium/Deep) x havuzlama (GAP/Flatten)
        "Shallow_GAP":  (SHALLOW, True,  "ConvNet"),
        "Shallow_Flat": (SHALLOW, False, "ConvNet"),
        "Medium_GAP":   (MEDIUM,  True,  "ConvNet"),
        "Medium_Flat":  (MEDIUM,  False, "ConvNet"),
        "Deep_GAP":     (DEEP,    True,  "ConvNet"),
        "Deep_Flat":    (DEEP,    False, "ConvNet"),
        # --- yüksek kapasiteli referans
        "ResNet18":     (None,    True,  "ResNet18"),
        # --- eski skaler geç füzyon (v1 ile kıyas için; v2'de aux kanalları tercih edilir)
        "MTGConvNet_LateFusion": (DEEP, False, "MTGConvNet_LateFusion"),
        "ResNet18_LateFusion":   (None, False, "ResNet18_LateFusion"),
    }
    ARCH_ABLATION = ["Shallow_GAP", "Shallow_Flat", "Medium_GAP", "Medium_Flat", "Deep_GAP", "Deep_Flat"]
    ALIASES = {  # kısa ad -> deney adı listesi
        "all": list(all_experiments),
        "arch": ARCH_ABLATION,                       # 6 mimari varyantı (danışman istediği kıyas)
        "arch_flat": ["Shallow_Flat", "Medium_Flat", "Deep_Flat"],
        "mtg_flat": ["Deep_Flat"],                   # geriye uyumluluk
        "shallow_flat": ["Shallow_Flat"], "medium_flat": ["Medium_Flat"], "deep_flat": ["Deep_Flat"],
        "shallow_gap": ["Shallow_GAP"], "medium_gap": ["Medium_GAP"], "deep_gap": ["Deep_GAP"],
        "resnet18": ["ResNet18"],
        "mtg_late_fusion": ["MTGConvNet_LateFusion"], "resnet18_fusion": ["ResNet18_LateFusion"],
    }
    if args.model in ALIASES:
        names = ALIASES[args.model]
    elif args.model in all_experiments:
        names = [args.model]
    else:
        raise SystemExit(f"Bilinmeyen --model '{args.model}'. Seçenekler: "
                         + ", ".join(sorted(set(list(ALIASES) + list(all_experiments)))))
    experiments = {n: all_experiments[n] for n in names}
    print(f"Çalıştırılacak deney(ler): {', '.join(experiments)}")
    
    # Initialize Dataloaders ONCE
    print("Initializing Datasets and DataLoaders...")
    test_loaders = None
    if args.split:
        loaders = get_split_dataloaders(h5_path, args.split, batch_size=BATCH_SIZE, num_workers=n_workers,
                                        use_aux=args.aux, aux_channels=aux_channels, shuffle_train=False)
        train_loader, val_loader = loaders["train"], loaders["val"]
        test_loaders = {k: v for k, v in loaders.items() if k.startswith("test")}
    elif os.path.exists(h5_path):
        print(f"HDF5 dataset found at {h5_path}. Using fast H5 loading!")
        train_loader, val_loader = get_dataloaders(
            h5_path=h5_path,
            batch_size=BATCH_SIZE,
            num_workers=n_workers,
            shuffle=False,  # Must be False for fast sequential HDF5 reads. Shuffling is handled by buffered_shuffle_generator.
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
            num_workers=n_workers,
            shuffle=IS_COLAB
        )
    
    
    # Load existing results to support resuming from disconnects
    import json
    json_path = args.results or out_path("ablation_results.json")
    results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
            print(f"Found existing results for {list(results.keys())}. Resuming from last unfinished model.")
        except Exception as e:
            print(f"Could not load existing results: {e}. Starting fresh.")
    
    # Automatically run 3 epochs and 30 batches locally for a meaningful dry-run
    # Set limit_batches = None to train on the entire dataset.
    epochs_to_run = args.epochs if args.epochs is not None else (3 if not IS_COLAB else 15)
    limit_batches = args.limit_batches if args.limit_batches is not None else (30 if not IS_COLAB else None)
    
    for name, config in experiments.items():
        blocks, use_gap, model_type = config
        if name in results:
            print(f"Model '{name}' already trained. Skipping.")
            continue
            
        history, best_f1, num_params, test_results = run_experiment(
            name, 
            blocks, 
            train_loader,
            val_loader,
            epochs=epochs_to_run, 
            limit_batches=limit_batches,
            use_gap=use_gap,
            model_type=model_type,
            test_loaders=test_loaders
        )
        results[name] = {
            "history": history,
            "best_f1": best_f1,            # = en iyi val balanced accuracy
            "best_val_balanced_accuracy": best_f1,
            "num_params": num_params,
            "test": test_results,
            "config": {"seed": args.seed, "aux": args.aux, "aux_channels": aux_channels, "split": args.split,
                       "epochs": epochs_to_run, "limit_batches": limit_batches}
        }
        
        # Save results immediately to prevent data loss on disconnect
        try:
            atomic_json(results, json_path)
            print(f"'{name}' sonuçları kaydedildi -> {json_path}")
        except Exception as e:
            print(f"Warning: could not save incremental results to {json_path}: {e}")
        
    # Print summary comparative table
    print("\n" + "="*50)
    print("         ABLATION STUDY COMPARATIVE RESULTS")
    print("="*50)
    print(f"{'Model Name':<22} | {'Params':<10} | {'Val BalAcc':<10} | {'Test_time BalAcc/MCC':<22} | {'Test_station BalAcc/MCC':<22}")
    print("-"*100)
    for name, res in results.items():
        t = res.get("test", {}) or {}
        tt = t.get("test_time", {}); ts = t.get("test_station", {})
        f = lambda m: f"{m['balanced_accuracy']:.4f}/{m['mcc']:.4f}" if m else "-"
        print(f"{name:<22} | {res['num_params']:<10,} | {res['best_f1']:<10.4f} | {f(tt):<22} | {f(ts):<22}")
    
    print(f"\nAblation study results successfully finalized at {json_path}")
    
    # Eski tekil grafik betiği yalnızca varsayılan artifacts/ablation_results.json ile çalışır.
    # Toplu raporlama artık src/aggregate_results.py ile yapılıyor; bu yüzden yalnızca
    # varsayılan yol kullanıldığında çağrılır (aksi halde her koşuda anlamsız hata basıyordu).
    if os.path.abspath(json_path) == os.path.abspath(os.path.join("artifacts", "ablation_results.json")):
        import subprocess
        try:
            subprocess.run([sys.executable, os.path.join("src", "plot_results.py")], cwd=os.getcwd(), check=True)
        except Exception as e:
            print(f"Grafik üretilemedi: {e}")
    else:
        print(f"Rapor için: python src/aggregate_results.py --dir {os.path.dirname(json_path) or '.'} --plot")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Exception in main execution:")
        traceback.print_exc()
        exit(1)
