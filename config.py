import os, torch, pickle
import numpy as np
import shutil
import wandb

# Check in 'corradini' is in the path
if 'corradini' in os.getcwd():
    USERNAME = "Barbara"
    wandb.login(key="f904ed1462c53edef7fef2f82b6c04e99ea34339")
elif 'giulio' in os.getcwd():
    wandb.login(key="03c8065e56dc0abe77944c0bcfbc88b313878717")
    USERNAME = "Alessia"
    
WANDB_ENTITY_NAME = "ginestra"
WANDB_PROJECT_NAME = f"GINESTRA_nuovo"

# Remove __pycache__ folder if it exists
if os.path.exists(os.path.join(os.path.dirname(__file__), "__pycache__")):
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "__pycache__"))
if os.path.exists(os.path.join(os.path.dirname(__file__), "models/__pycache__")):
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "models/__pycache__"))

## === FILESYSTEM PARAMETERS === ##

# Set the base directory and data directory
BASEDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(BASEDIR, "data/data")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## === END FILESYSTEM PARAMETERS === ##

## === TRAINING EXPERIMENTAL PARAMETERS === ##

N_EPOCHS = 100                  # Number of epochs for training
GRID_N_EPOCHS = 500             # Number of epochs for grid search
PARAM_GRID = {
    'dim_h': [512], 
    'drop_rate': [0.3],
    'learning_rate': [1e-4],
    'l2_rate': [1e-5], 
    'n_heads': [4],
}
N_RUNS = 5  # Number of runs for the model

## === END TRAINING EXPERIMENTAL PARAMETERS === ##

## === DATASET PARAMETERS === ##
# Target type for the dataset
# Options: "pathway", "superclass", "class"
# "pathway" = 7 classes
# "superclass" = 70 classes
# "class" = 652 classes
TARGET_TYPE = "class"  # Options: "pathway", "superclass", "class"

## DATASET PARAMETERS
FORCE_DATASET_GENERATION = True # If True, force the generation of the dataset
N_SAMPLES = None  # Number of samples to pick from the training set. If set to None, all samples are used
BATCH_SIZE = 32  # Batch size
RANDOMIZE_SAMPLES = True # Randomize the order of the samples in the dataset
USE_MULTILABEL = True # If True, use multilabel classification
TRAINING_SPLIT = 0.6  # Percentage of samples to use for training
VALIDATION_SPLIT = 0.2  # Percentage of samples to use for validation
# TEST_SPLIT = 0.2  # Percentage of samples to use for testing (automaticlly calculated)

CLS_LIST = None         # If None, all targets values are used (see TARGET_TYPE) otherwise CLS_LIST = [3, 6, PATHWAYS["Carbohydrates"], PATHWAYS["Amino acids and Peptides"]]   # Class labels of the dataset to be kept in training, validation and test sets

## DATASET ENCODING
USE_FINGERPRINT = False

## ATOM FEATURES (Atom symbols always present in "label" format)
USE_CHIRALITY = False               # (4 bits) A
USE_HYDROGENS_IMPLICIT = True       # (6 bits) B  T
USE_TOPOLOGICAL_FEATURES = True     # (6 bits) C  T
USE_CHARGE_PROPERTIES = True        # (1 int)  D Electronic / Charge Properties   T
USE_HYBRIDIZATION = True            # (7 ints) E  T
USE_RING_INFO = True                # (2 ints) F Ring and Aromaticity Information T
USE_ATOMIC_PROPERTIES = False       # (3 ints) G 

DATASET_ID = "" 

DATASET_ID += "A" if USE_CHIRALITY else ""
DATASET_ID += "B" if USE_HYDROGENS_IMPLICIT else ""
DATASET_ID += "C" if USE_TOPOLOGICAL_FEATURES else ""
DATASET_ID += "D" if USE_CHARGE_PROPERTIES else ""
DATASET_ID += "E" if USE_HYBRIDIZATION else ""
DATASET_ID += "F" if USE_RING_INFO else ""
DATASET_ID += "G" if USE_ATOMIC_PROPERTIES else "" 
# Sort the dataset ID
DATASET_ID = "" if not USE_MULTILABEL else "MULTILABEL_" + "".join(sorted(DATASET_ID))

# Write a dictionary of atom features to a file
ATOM_FEATURES_DICT = {
    "chirality": USE_CHIRALITY,
    "hydrogens_implicit": USE_HYDROGENS_IMPLICIT,
    "topological_features": USE_TOPOLOGICAL_FEATURES,
    "charge_properties": USE_CHARGE_PROPERTIES,
    "hybridization": USE_HYBRIDIZATION,
    "ring_info": USE_RING_INFO,
    "atomic_properties": USE_ATOMIC_PROPERTIES
}

## === END DATASET PARAMETERS === ##

## === EARLY STOPPING PARAMETERS == ##
EARLY_PATIENCE = 7
EARLY_MIN_DELTA = 0.0001
## === END EARLY STOPPING PARAMETERS == ##

## === EXPERIMENT PARAMETERS === ##

# Build dictionaries of classes, superclasses and pathways based on the target type
if TARGET_TYPE == "pathway":
    with open(f'{DATADIR}/char2idx_path_V1.pkl','rb') as f:
        class_  = pickle.load(f)
elif TARGET_TYPE == "superclass" or TARGET_TYPE == "super_class":
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

## === TELEGRAM PARAMETERS === ##
TOKEN = "7544529562:AAHiLMgvUXanJDWHm0xFOsIfbasBgEqOaQw"
CHAT_ID = "252746684"
## === END TELEGRAM PARAMETERS === ##