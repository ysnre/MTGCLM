import os
import sys

# Ensure Windows PATH includes conda environment's Library/bin for DLL resolution (matplotlib, numpy, etc.)
if sys.platform == "win32":
    env_dir = os.path.dirname(sys.executable)
    lib_bin = os.path.join(env_dir, "Library", "bin")
    if os.path.exists(lib_bin) and lib_bin not in os.environ["PATH"]:
        os.environ["PATH"] = lib_bin + os.pathsep + os.environ["PATH"]

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    json_path = os.path.join("artifacts", "ablation_results.json")
    if not os.path.exists(json_path):
        print(f"Error: JSON results file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        results = json.load(f)

    # Automatically map colors and linestyles to prevent collisions
    colormap = plt.get_cmap("tab10")
    
    # Plot Validation Loss
    plt.figure(figsize=(12, 7))
    for idx, (name, res) in enumerate(results.items()):
        val_loss = res["history"]["val_loss"]
        color = colormap(idx % 10)
        linestyle = "--" if "Flat" in name else "-"
        marker = "s" if "ResNet" in name else ("o" if "GAP" in name else "^")
        
        plt.plot(
            val_loss, 
            label=f"{name} Val Loss", 
            color=color, 
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0
        )
        
    plt.title("Validation Loss Comparison between Architectures")
    plt.xlabel("Epochs (1-indexed)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join("artifacts", "ablation_study_loss.png")
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Comparative loss plot successfully saved to {plot_path}")

    # Plot Validation F1-Score
    plt.figure(figsize=(12, 7))
    for idx, (name, res) in enumerate(results.items()):
        val_f1 = res["history"]["val_f1"]
        color = colormap(idx % 10)
        linestyle = "--" if "Flat" in name else "-"
        marker = "s" if "ResNet" in name else ("o" if "GAP" in name else "^")
        
        plt.plot(
            val_f1, 
            label=f"{name} Val F1-Score", 
            color=color, 
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0
        )
        
    plt.title("Validation F1-Score Comparison between Architectures")
    plt.xlabel("Epochs (1-indexed)")
    plt.ylabel("F1-Score")
    # Let matplotlib auto-scale the y-axis to dynamically capture the true range
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    f1_plot_path = os.path.join("artifacts", "ablation_study_f1.png")
    plt.savefig(f1_plot_path, dpi=150)
    plt.close()
    print(f"Comparative F1 plot successfully saved to {f1_plot_path}")

if __name__ == "__main__":
    main()
