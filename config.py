"""Configuration for self-correction experiments on reasoning models."""

# MODEL CONFIG
# Neel Nanda recommends R1 distilled Qwen 1.5B for studying reasoning models
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# DATASET CONFIG
DATASET_NAME = "gsm8k"
N_PROBLEMS = 25

# EXPERIMENT CONFIG
EDIT_TYPES = ["factual", "style", "contradict", "random"]
POSITION_FRACTIONS = [0.2, 0.5, 0.8]

# GENERATION CONFIG
MAX_COT_TOKENS = 512  # R1 produces longer CoT natively
MAX_CONTINUATION_TOKENS = 300
TEMPERATURE = 0.7

# CORRECTION DETECTION
EXPLICIT_CORRECTION_PHRASES = [
    "wait",
    "actually",
    "no,",
    "that's wrong",
    "let me reconsider",
    "hold on",
    "hmm",
    "let me think again",
]

# OUTPUT
RESULTS_FILE = "results.json"
PLOTS_DIR = "plots"

# CHECKPOINTING
CHECKPOINT_EVERY = 5  # save partial results every N problems
CHECKPOINT_PATH = "results_checkpoint.json"

