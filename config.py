import os, torch
import numpy as np

REPRODUCIBLE = True
if REPRODUCIBLE:
    SEED = 123
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
        
# Set the base directory and data directory
BASEDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(BASEDIR, "data")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EPOCHS = 10
N_RUNS = 3  # Number of runs for the model
## DATASET PARAMETERS
USE_AVAILABLE_DATASET = False # If True, use the dataset already downloaded and preprocessed
N_SAMPLES = None  # Number of samples to pick from the training set. If set to None, all samples are used
BATCH_SIZE = 32  # Batch size
RANDOMIZE_SAMPLES = True # Randomize the order of the samples in the dataset
MULTILABEL2MULTICLASS = False

# CLS_LIST = [3, 6, PATHWAYS["Carbohydrates"], PATHWAYS["Amino acids and Peptides"]]   # Class labels of the dataset to be kept in training, validation and test sets
CLS_LIST = None         # If None, all targets values are used (see TARGET_TYPE),
TARGET_TYPE = "pathway"  # Options: "pathway", "superclass", "class"

## DATASET ENCODING
TARGET_MODE = "hot" # if CLS_LIST is not None and len(CLS_LIST) > 2 else "binary" # Options: "binary" or "ohe" (one-hot encoding)
USE_FINGERPRINT = False

## NETWORK CONFIG
H_DIM = 128
MODELS = ["gin", "gine"] #, "mlp"] # Options: "gin", "gine", "mlp"
# OR
# MODELS = ["mlp"] 

import pickle
# Build dictionaries of classes, superclasses and pathways based on the target type
if TARGET_TYPE == "pathway":
    with open(f'{DATADIR}/char2idx_path_V1.pkl','rb') as f:
        class_  = pickle.load(f)
elif TARGET_TYPE == "superclass":
    with open(f'{DATADIR}/char2idx_super_V1.pkl','rb') as f:
        class_  = pickle.load(f)
elif TARGET_TYPE == "class":
    with open(f'{DATADIR}/char2idx_class_V1.pkl','rb') as f:
        class_  = pickle.load(f)
else:
    raise ValueError("TARGET_TYPE must be one of 'pathway', 'superclass' or 'class'")

PATHWAYS = {k: v for k, v in class_.items()}
# LABELS_CODES in one-hot encoding
LABELS_CODES = {i: np.array([1 if i == j else 0 for j in range(len(class_))]) for i in range(len(class_))}

if MULTILABEL2MULTICLASS and TARGET_TYPE == "pathway":
    PATHWAYS["Amino acids and Peptides Polyketides"] = 7
    PATHWAYS["Alkaloids Terpenoids"] = 8
    PATHWAYS["Polyketides Terpenoids"] = 9
    LABELS_CODES[7] = np.array([0,1,0,0,1,0,0])
    LABELS_CODES[8] = np.array([1,0,0,0,0,0,1])
    LABELS_CODES[9] = np.array([0,0,0,0,1,0,1])