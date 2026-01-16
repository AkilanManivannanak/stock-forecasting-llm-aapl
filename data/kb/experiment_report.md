# AAPL Forecasting Experiment Report

## Model performance (full test split)

| model         |     rmse |
|:--------------|---------:|
| naive         |  3.53333 |
| ma_5          |  5.60702 |
| lstm          | 15.0827  |
| random_forest | 39.1563  |

Best model on RMSE: **naive** with RMSE ≈ 3.533.

## Plots

- `plots/actual_vs_pred_lstm.png` – Actual vs LSTM predicted close prices on the held-out test set.

## Notes

- Baselines are intentionally included; naive can be hard to beat.
