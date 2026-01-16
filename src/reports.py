import os
import pandas as pd

def write_experiment_report(metrics_csv: str, out_path: str):
    metrics = pd.read_csv(metrics_csv).sort_values("rmse")
    best = metrics.iloc[0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        f.write("# AAPL Forecasting Experiment Report\n\n")
        f.write("## Model performance (full test split)\n\n")
        f.write(metrics.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"Best model on RMSE: **{best['model']}** with RMSE ≈ {best['rmse']:.3f}.\n\n")
        f.write("## Plots\n\n")
        f.write("- `plots/actual_vs_pred_lstm.png` – Actual vs LSTM predicted close prices on the held-out test set.\n\n")
        f.write("## Notes\n\n")
        f.write("- Baselines are intentionally included; naive can be hard to beat.\n")
