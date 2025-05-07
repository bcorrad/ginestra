import os, torch, pickle
import numpy as np

REPRODUCIBLE = True

## === FILESYSTEM PARAMETERS === ##

# Set the base directory and data directory
BASEDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(BASEDIR, "data")

# Dataset information file
DATASET_INFO_FILE = os.path.join(DATADIR, "dataset_info.csv")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## === END FILESYSTEM PARAMETERS === ##

## === TRAINING EXPERIMENTAL PARAMETERS === ##

N_EPOCHS = 100            # Number of epochs for training
GRID_N_EPOCHS = 100     # Number of epochs for grid search
PARAM_GRID = {
    'dim_h': [16, 64, 128],
    'drop_rate': [0.1, 0.3, 0.5],
    'learning_rate': [1e-4],
    'l2_rate': [5e-4],
    'n_heads': [2, 4],
}
N_RUNS = 3  # Number of runs for the model

## === END TRAINING EXPERIMENTAL PARAMETERS === ##

## === DATASET PARAMETERS === ##

## DATASET PARAMETERS
USE_AVAILABLE_DATASET = None # If True, use the dataset already downloaded and preprocessed
FORCE_DATASET_GENERATION = False # If True, force the generation of the dataset
N_SAMPLES = None  # Number of samples to pick from the training set. If set to None, all samples are used

BATCH_SIZE = 32  # Batch size
RANDOMIZE_SAMPLES = True # Randomize the order of the samples in the dataset
MULTILABEL2MULTICLASS = False

TRAINING_SPLIT = 0.6  # Percentage of samples to use for training
VALIDATION_SPLIT = 0.2  # Percentage of samples to use for validation
# TEST_SPLIT = 0.2  # Percentage of samples to use for testing (automaticlly calculated)

# CLS_LIST = [3, 6, PATHWAYS["Carbohydrates"], PATHWAYS["Amino acids and Peptides"]]   # Class labels of the dataset to be kept in training, validation and test sets
CLS_LIST = None         # If None, all targets values are used (see TARGET_TYPE),
TARGET_TYPE = "superclass"  # Options: "pathway", "superclass", "class"

## DATASET ENCODING
TARGET_MODE = "hot" # if CLS_LIST is not None and len(CLS_LIST) > 2 else "binary" # Options: "binary" or "ohe" (one-hot encoding)
USE_FINGERPRINT = False

## ATOM FEATURES
USE_CHIRALITY = False           # (4 bits) A
USE_HYDROGENS_IMPLICIT = False  # (6 bits) B
USE_TOPOLOGICAL_FEATURES = True # (6 bits) C
USE_CHARGE_PROPERTIES = True    # (1 int) D
USE_HYBRIDIZATION = True        # (7 ints) E 
USE_RING_INFO = True            # (2 ints) F 
USE_ATOMIC_PROPERTIES = True    # (3 ints) G 

DATASET_ID = ""

DATASET_ID += "A" if USE_CHIRALITY else ""
DATASET_ID += "B" if USE_HYDROGENS_IMPLICIT else ""
DATASET_ID += "C" if USE_TOPOLOGICAL_FEATURES else ""
DATASET_ID += "D" if USE_CHARGE_PROPERTIES else ""
DATASET_ID += "E" if USE_HYBRIDIZATION else ""
DATASET_ID += "F" if USE_RING_INFO else ""
DATASET_ID += "G" if USE_ATOMIC_PROPERTIES else "" 
# Sort the dataset ID
DATASET_ID = "".join(sorted(DATASET_ID))

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

## === NETWORK PARAMETERS === ##

## NETWORK CONFIG
H_DIM = 16
# MODELS = ["gine", "gin", "gat", "gate", "mlp"] # Options: "gin", "gine", "mlp", "gat", "gate"
# OR
MODELS = ["mlp"] # Only for non-grid search setup
MODELS.sort()  # Minimize the dataset exchanges between models during training

## === END NETWORK PARAMETERS === ##

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

if MULTILABEL2MULTICLASS and TARGET_TYPE == "pathway":
    PATHWAYS["Amino acids and Peptides Polyketides"] = 7
    PATHWAYS["Alkaloids Terpenoids"] = 8
    PATHWAYS["Polyketides Terpenoids"] = 9
    LABELS_CODES[7] = np.array([0,1,0,0,1,0,0])
    LABELS_CODES[8] = np.array([1,0,0,0,0,0,1])
    LABELS_CODES[9] = np.array([0,0,0,0,1,0,1])
    
## === END EXPERIMENT PARAMETERS === ##
    
# # Initialize experiment folder 
# import datetime
# EXPERIMENT_FOLDER = os.path.join(BASEDIR, "experiments", "-".join(MODELS) + "_" + TARGET_TYPE.lower() + "_" +datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
# os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)

# # Create a models folder
# os.makedirs(os.path.join(EXPERIMENT_FOLDER, "models"), exist_ok=True)
# os.makedirs(os.path.join(EXPERIMENT_FOLDER, "utils"), exist_ok=True)
# # Create the weights folder
# os.makedirs(os.path.join(EXPERIMENT_FOLDER, "pt"), exist_ok=True)
# # Create the reports folder
# os.makedirs(os.path.join(EXPERIMENT_FOLDER, "reports"), exist_ok=True)
# # Create plots folder
# os.makedirs(os.path.join(EXPERIMENT_FOLDER, "plots"), exist_ok=True)

# # Save the configuration file
# import shutil
# shutil.copy(__file__, os.path.join(EXPERIMENT_FOLDER, "config.py"))

# # Save the models file from the directory
# for model_file in os.listdir(os.path.join(BASEDIR, "models")):
#     if model_file.endswith(".py"):
#         shutil.copy(os.path.join(BASEDIR, "models", model_file), os.path.join(EXPERIMENT_FOLDER, "models", model_file))
        
# # Copy all the py files in the utils folder
# for file in os.listdir(os.path.join(BASEDIR, "utils")):
#     if file.endswith(".py"):
#         shutil.copy(os.path.join(BASEDIR, "utils", file), os.path.join(EXPERIMENT_FOLDER, "utils", file))
        
# # Copy all the py files in the folder 
# for file in os.listdir(BASEDIR):
#     if file.endswith(".py") and file != "config.py":
#         shutil.copy(os.path.join(BASEDIR, file), os.path.join(EXPERIMENT_FOLDER, file))