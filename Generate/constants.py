# Centralized constants and defaults for blueprint generation

VERSION = "v1"

# Generation defaults
MIN_SEPARATION_DEFAULT = 1.0
DECODE_MAX_ATTEMPTS = 2
GUIDED_TOPK_DEFAULT = 8
GUIDED_BEAM_DEFAULT = 8
TEMPERATURE_DEFAULT = 1.0
BEAM_SIZE_DEFAULT = 5
STRATEGY_DEFAULT = "guided"  # one of: greedy, sample, beam, guided
BACKEND_DEFAULT = "auto"     # one of: auto, cp, model

# Packing and validation
GRID_DEFAULT = 1.0
ZONING_DEFAULT = True
MIN_HALL_WIDTH_DEFAULT = 4.0

# CP-SAT
CPSAT_TIME_LIMIT_S = 8.0

# Checkpoint
CKPT_FILENAME = "model_latest.pth"
