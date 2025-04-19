from typing import Literal
import os
import datetime
import numpy as np

def initialize_experiment(
    models: Literal["mlp", "gin", "gine", "gat", "gate"],
    target_type: Literal["pathway", "superclass", "class"],
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    ):
    
    MODELS = models
    TARGET_TYPE = target_type
    BASEDIR = base_dir
        
    # Initialize experiment folder 
    EXPERIMENT_FOLDER = os.path.join(BASEDIR, "experiments", MODELS + "_" + TARGET_TYPE.lower() + "_" +datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)

    # Create a models folder
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "models"), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "utils"), exist_ok=True)
    # Create the weights folder
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "pt"), exist_ok=True)
    # Create the reports folder
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "reports"), exist_ok=True)

    # Save the configuration file
    import shutil
    shutil.copy(__file__, os.path.join(EXPERIMENT_FOLDER, "config_gridsearch.py"))

    # Save the models file from the directory
    for model_file in os.listdir(os.path.join(BASEDIR, "models")):
        if model_file.endswith(".py"):
            shutil.copy(os.path.join(BASEDIR, "models", model_file), os.path.join(EXPERIMENT_FOLDER, "models", model_file))
            
    # Copy all the py files in the utils folder
    for file in os.listdir(os.path.join(BASEDIR, "utils")):
        if file.endswith(".py"):
            shutil.copy(os.path.join(BASEDIR, "utils", file), os.path.join(EXPERIMENT_FOLDER, "utils", file))
            
    # Copy all the py files in the folder 
    for file in os.listdir(BASEDIR):
        if file.endswith(".py") and file != "config.py":
            shutil.copy(os.path.join(BASEDIR, file), os.path.join(EXPERIMENT_FOLDER, file))
            
    return EXPERIMENT_FOLDER