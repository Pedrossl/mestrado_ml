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

FEATURE_DROP_COLUMNS = [
    "Poverty Status",
    "Number of Siblings",
    "Family History - Psychiatric Diagnosis",
    "Number of Bio. Parents",
]

FEATURE_DROP_RATIONALE = {
    "Poverty Status": "Correlacao Spearman quase nula com GAD e redundancia com Race/Number of Bio. Parents.",
    "Number of Siblings": "Correlacao Spearman quase nula com GAD e baixa importancia no XGBoost.",
    "Family History - Psychiatric Diagnosis": "Correlacao Spearman fraca com GAD e baixa importancia no XGBoost.",
    "Number of Bio. Parents": "Correlacao Spearman fraca com GAD, VIF moderado e redundancia com Race/Poverty Status.",
}

SENSITIVE_FEATURES = [
    "Race",
    "Sex",
    "Poverty Status",
    "Number of Bio. Parents",
]
