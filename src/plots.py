import os
import numpy as np
import matplotlib.pyplot as plt

def plot_actual_vs_pred(y_true, y_pred, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    plt.figure(figsize=(16, 6))
    plt.plot(y_true, label="Actual Close", linewidth=1.5)
    plt.plot(y_pred, label="LSTM Predicted Close", linewidth=2, alpha=0.8)

    error = np.abs(y_true - y_pred)
    plt.fill_between(np.arange(len(y_true)), y_true - error, y_true + error, alpha=0.08, label="|Error band|")

    plt.title("Actual vs Predicted (LSTM) – Test Set", fontsize=16)
    plt.xlabel("Test Time Index (days)")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")

    out_path = os.path.join(plots_dir, "actual_vs_pred_lstm.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")
