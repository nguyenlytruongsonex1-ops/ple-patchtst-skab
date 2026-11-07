# SKAB-only, historical budget + RevIN quantile PLE

Explicit local protocol: `python -m exp.skab_oldbudget.run --output /workspace/results/UNIQUE`.
CPU, one Torch thread, sequential 5 folds x seeds 0..4, 30 epochs without early stopping.
Best validation AP checkpoint; threshold maximizing validation F1. No test tuning.
No TabM/plain/Bike runs. Historical values stay untouched.

Uses pilot RevIN + quantile PLE implementation (train-window pre-affine RevIN edges,
8 bins, per-channel zero-initialized residual). Old PatchTST architecture and
training budget retained, including actual BatchNorm encoder behavior.
Raw data checked against the existing 35-file official manifest; grouped split,
train-only StandardScaler, windows and anomaly end-label unchanged. Exact split,
raw, scaler, window, label, source/config/diff hashes saved with results.

`patience=None` is explicit no-early-stop support; pilot defaults remain unchanged.
Run 18-test correctness suite first. Its combined official-data integration test
requires Bike too and skips without it; SKAB hashes are independently asserted.
Historical comparison is descriptive: RevIN and bin construction both change,
and historical runtime hardware differs. Do not attribute change to bins alone.
