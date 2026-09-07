from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_DATASET = DATASETS_DIR / "mestrado-treino.csv"
TEST_DATASET = DATASETS_DIR / "mestrado-teste.csv"

RANDOM_STATE = 42
N_SPLITS = 10
TARGETS = ("GAD", "SAD")

NUMERIC_COLUMNS = [
    "Age",
    "Number of Impairments",
    "Number of Type A Stressors",
    "Number of Type B Stressors",
    "Frequency Temper Tantrums",
    "Frequency Irritable Mood",
    "Number of Sleep Disturbances",
    "Number of Physical Symptoms",
    "Number of Sensory Sensitivities",
]

PREPROCESSING_DROP_COLUMNS = [
    "Depression",
    "Number of Type A Stressors",
    "Number of Physical Symptoms",
    "Family History - Substance Abuse",
]

MODEL_DROP_COLUMNS = [
    "Subject",
    "GAD Probabiliy - Gamma",
    "SAD Probability - Gamma",
    "Sample Weight",
]

SENSITIVE_FEATURES = [
    "Race",
    "Sex",
    "Poverty Status",
    "Number of Bio. Parents",
]
