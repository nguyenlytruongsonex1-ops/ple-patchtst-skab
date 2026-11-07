"""Fixed pilot hyperparameters.

Shared by plain PatchTST, PatchTST-PLE-quantile, and TabM.
Nothing here was chosen by looking at test scores of the new models.
"""

from __future__ import annotations

N_BINS = 8
# In RevIN z-score units. Narrower than this is treated as a tied edge.
WIDTH_EPS = 1e-8
REVIN_EPS = 1e-5

PATCH_LEN = 8
PATCH_STRIDE = 4
D_MODEL = 32
N_HEADS = 4
D_FF = 64
N_LAYERS = 1
# Passed through for call compatibility with run_detect.py. The vendored
# backbone does not forward this argument, so the encoder stays BatchNorm1d.
NORM = "LayerNorm"
PADDING_PATCH = "end"

DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BATCH_SIZE = 256

# Pilot budget. Full 5 fold x 5 seed is a later decision.
MAX_EPOCHS = 10
PATIENCE = 3
SEEDS = (0, 1)
SKAB_FOLDS = (0, 1)

SKAB_LOOKBACK = 60  # steps on the 1-second grid, not a claim of uniform physical seconds
SKAB_WINDOW_STRIDE = 4

BIKE_LOOKBACK = 168  # actual clock hours
BIKE_HORIZON = 24
BIKE_TRAIN_FRAC = 0.70
BIKE_VAL_FRAC = 0.15  # val ends at train_frac + val_frac of the complete hourly grid

# Same explicit TabM.make arguments as run_detect.py (tabm==0.0.3).
TABM_BLOCKS = 2
TABM_D_BLOCK = 64
TABM_K = 16

PILOT_MODELS = ("PatchTST", "PatchTST-PLE-quantile", "TabM")
# Kept in run_detect.py. Not part of this pilot.
EQUAL_BIN_MODEL = "PatchTST-PLE-input"

MAX_PILOT_RUNS = 18
PREDICT_BATCH = 512
