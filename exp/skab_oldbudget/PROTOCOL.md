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

## Authorized no-RevIN ablation (2026-11-07)
Use `--no-revin` to run only quantile PLE without RevIN. Default is still RevIN on.
StandardScaler stays train-only and unchanged. Fit 8-bin knots on TRAIN windows
in global-scaled coordinates (not on raw units or RevIN outputs). No reused RevIN
knots; PLE input and knot coordinates match. No RevIN affine parameters when off:
13,218 vs 13,234 parameters, a difference of 16 for 8 channels.

All 25 old-budget jobs, splits, seed values, optimizer, batching, checkpoint and
threshold rules remain fixed. Tests verify all common initial weights agree for
seeds 0..4 on/off, and zero-PLE/no-RevIN matches the old plain PatchTST exactly.
RevIN uses constant initialization, consuming no RNG; the default RevIN-on path
was additionally compared bitwise against afe8b7d weights and predictions.
This is a whole RevIN-pipeline ablation, including different quantile coordinates
and the affine removal; it does not keep literal encodings identical.

Compare descriptively to the existing 25 RevIN-on runs, pairing fold+seed labels,
then averaging seeds within each fold. Fold SD uses ddof=1. Do not treat 25 runs
as independent datasets or claim harm on all tasks; folds share training data.
No baseline retraining, test tuning, seed selection, or model expansions.
`progress.json` gives completed count and observed-time remaining estimate.
