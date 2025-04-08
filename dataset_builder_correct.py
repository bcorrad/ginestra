import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 
from utils.utils import select_class_idx_path, select_class_idx_super
import torch
from config import MODEL
from fingerprint_handler import calculate_fingerprint
import os

from torch_geometric.loader import DataLoader
from utils.graph_data_def import create_pytorch_geometric_graph_data_list_from_smiles_and_labels
from utils.utils import data_generation

from config import DATADIR, N_SAMPLES, BATCH_SIZE, RANDOMIZE_SAMPLES, CLS_LIST, TARGET_MODE, TARGET_TYPE, USE_FINGERPRINT
np.random.seed(42)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

with open(f'{DATADIR}/char2idx_class_V1.pkl','rb') as f:
    class_  = pickle.load(f)
with open(f'{DATADIR}/char2idx_super_V1.pkl','rb') as f:
    superclass_  = pickle.load(f)
with open(f'{DATADIR}/char2idx_path_V1.pkl','rb') as f:
    pathway_  = pickle.load(f)
with open(f'{DATADIR}/datset_class_all_V1.pkl','rb') as r:
    dataset = pickle.load(r)

# Train, Validation, and test set 
molecule_inchikey = list(dataset.keys())
np.random.shuffle(molecule_inchikey)
dict_ = np.array(molecule_inchikey)
Y_ = np.array([np.max(np.where(dataset[i]['Class']==1)[0]) for i in dict_]) # The maximum index of the class array where the value is 1

train_D, test_dict, y_train, y_test = train_test_split(dict_, Y_, test_size=0.2, random_state=1, stratify=Y_)
train_dict, val_dict, y_train, y_val = train_test_split(train_D, y_train, test_size=0.2, random_state=1, stratify=y_train)


if os.path.exists(f'{DATADIR}/train_dataset_w_fingerprints.pkl') \
    or os.path.exists(f'{DATADIR}/val_dataset_w_fingerprints.pkl') \
        or os.path.exists(f'{DATADIR}/test_dataset_w_fingerprints.pkl'):    
    
    print("Loading datasets...")
    train_dict = load_pickle(f'{DATADIR}/train_dataset_w_fingerprints.pkl')
    val_dict = load_pickle(f'{DATADIR}/val_dataset_w_fingerprints.pkl')
    test_dict = load_pickle(f'{DATADIR}/test_dataset_w_fingerprints.pkl')
    print("Datasets loaded successfully.")
    
    train_smiles, train_fing, train_path, train_super, train_class = train_dict['smiles'], train_dict['fingerprint'], train_dict['y_path'], train_dict['y_super'], train_dict['y_class']
    val_smiles, val_fing, val_path, val_super, val_class = val_dict['smiles'], val_dict['fingerprint'], val_dict['y_path'], val_dict['y_super'], val_dict['y_class']
    test_smiles, test_fing, test_path, test_super, test_class = test_dict['smiles'], test_dict['fingerprint'], test_dict['y_path'], test_dict['y_super'], test_dict['y_class']

else:
    print("Generating datasets...")
    # Generazione dei dataset
    train_smiles, train_fing, train_path, train_super, train_class = data_generation(train_dict, dataset, save_dataset=None)
    val_smiles, val_fing, val_path, val_super, val_class = data_generation(val_dict, dataset, save_dataset=None)
    test_smiles, test_fing, test_path, test_super, test_class  = data_generation(test_dict, dataset, save_dataset=None)

    # Save the datasets in a pkl file   
    train_dict = {'smiles': train_smiles, 'fingerprint': train_fing, 'y_path': train_path, 'y_super': train_super, 'y_class': train_class}
    val_dict = {'smiles': val_smiles, 'fingerprint': val_fing, 'y_path': val_path, 'y_super': val_super, 'y_class': val_class}
    test_dict = {'smiles': test_smiles, 'fingerprint': test_fing, 'y_path': test_path, 'y_super': test_super, 'y_class': test_class}

    with open(f'{DATADIR}/train_dataset_w_fingerprints.pkl', 'wb') as f:
        pickle.dump(train_dict, f)
    with open(f'{DATADIR}/val_dataset_w_fingerprints.pkl', 'wb') as f:
        pickle.dump(val_dict, f)
    with open(f'{DATADIR}/test_dataset_w_fingerprints.pkl', 'wb') as f:
        pickle.dump(test_dict, f)

# Selezione dei target in base al TARGET_TYPE    
if TARGET_TYPE == "pathway":
    print("Training")
    train_paths, train_smiles_pathway, train_finger = select_class_idx_path(train_path, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
    print("Validation")
    val_paths, val_smiles_pathway, val_fing = select_class_idx_path(val_path, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
    print("Test")
    test_paths, test_smiles_pathway, test_fing = select_class_idx_path(test_path, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
elif TARGET_TYPE == "superclass":
    print("Training")
    train_super, train_smiles_super, train_finger = select_class_idx_super(train_super, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
    print("Validation")
    val_super, val_smiles_super, val_fing = select_class_idx_super(val_super, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
    print('CONTROLLAREEEE')
    
    print("Test")
    test_super, test_smiles_super, test_fing = select_class_idx_super(test_super, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
elif TARGET_TYPE == "class":
    print("Training")
    train_class, train_smiles_class, train_finger = select_class_idx_super(train_class, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
    print("Validation")
    val_class, val_smiles_class, val_fing = select_class_idx_super(val_class, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
    print("Test")
    test_class, test_smiles_class, test_fing  = select_class_idx_super(test_class, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)

# train_smiles objects have to be shaped as 1-D arrays. Check the if the variables exist and if they are 1-D arrays
if TARGET_TYPE == "pathway":
    if len(train_smiles_pathway.shape) > 1:
        train_smiles_pathway = train_smiles_pathway.reshape(-1)
    if len(val_smiles_pathway.shape) > 1:
        val_smiles_pathway = val_smiles_pathway.reshape(-1)
    if len(test_smiles_pathway.shape) > 1:
        test_smiles_pathway = test_smiles_pathway.reshape(-1)
elif TARGET_TYPE == "superclass":
    if len(train_smiles_super.shape) > 1:
        train_smiles_super = train_smiles_super.reshape(-1)
    if len(val_smiles_super.shape) > 1:
        val_smiles_super = val_smiles_super.reshape(-1)
    if len(test_smiles_super.shape) > 1:
        test_smiles_super = test_smiles_super.reshape(-1)
elif TARGET_TYPE == "class":
    if len(train_smiles_class.shape) > 1:
        train_smiles_class = train_smiles_class.reshape(-1)
    if len(val_smiles_class.shape) > 1:
        val_smiles_class = val_smiles_class.reshape(-1)
    if len(test_smiles_class.shape) > 1:
        test_smiles_class = test_smiles_class.reshape(-1)

# Creazione dei DataLoader
if MODEL.lower() != "mlp":
    train_df = pd.DataFrame({
        'SMILES': train_smiles_pathway if TARGET_TYPE == "pathway" else train_smiles_super.tolist()
        if TARGET_TYPE == "superclass" else train_smiles_class.tolist(),
        'Labels': train_paths.tolist() if TARGET_TYPE == "pathway" else train_super.squeeze(1).tolist()
        if TARGET_TYPE == "superclass" else train_class.squeeze(1).tolist(),
        'Fingerprint': train_finger.squeeze(1).tolist() if USE_FINGERPRINT else None # n_BATCH x n_FEATURES
    })

    val_df = pd.DataFrame({
        'SMILES': val_smiles_pathway if TARGET_TYPE == "pathway" else val_smiles_super.tolist()
        if TARGET_TYPE == "superclass" else val_smiles_class.tolist(),
        'Labels': val_paths.tolist() if TARGET_TYPE == "pathway" else val_super.squeeze(1).tolist()
        if TARGET_TYPE == "superclass" else val_class.squeeze(1).tolist(),
        'Fingerprint': val_fing.squeeze(1).tolist() if USE_FINGERPRINT else None
    })

    test_df = pd.DataFrame({
        'SMILES': test_smiles_pathway if TARGET_TYPE == "pathway" else test_smiles_super.tolist()
        if TARGET_TYPE == "superclass" else test_smiles_class.tolist(),
        'Labels': test_paths.tolist() if TARGET_TYPE == "pathway" else test_super.squeeze(1).tolist()
        if TARGET_TYPE == "superclass" else test_class.squeeze(1).tolist(),
        'Fingerprint': test_fing.squeeze(1).tolist() if USE_FINGERPRINT else None
    })

    train_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(train_df)
    val_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(val_df)
    test_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(test_df)
    
    from torch_geometric.loader import DataLoader as GeoDataLoader

    train_dataloader = GeoDataLoader(train_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    val_dataloader = GeoDataLoader(val_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    test_dataloader = GeoDataLoader(test_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
else:
    from torch.utils.data import DataLoader, TensorDataset
    if TARGET_TYPE == 'pathway':
        train_dataset = TensorDataset(torch.Tensor(np.array(train_finger)).squeeze(1), torch.Tensor(np.array(train_paths)))
        val_dataset = TensorDataset(torch.Tensor(np.array(val_fing)).squeeze(1), torch.Tensor(np.array(val_paths)))    
        test_dataset = TensorDataset(torch.Tensor(np.array(test_fing)).squeeze(1), torch.Tensor(np.array(test_paths)))  
    elif TARGET_TYPE == 'superclass':
        train_dataset = TensorDataset(torch.Tensor(np.array(train_finger)).squeeze(1), torch.Tensor(np.array(train_super).squeeze(1))) # n_BATCH x n_FEATURES
        val_dataset = TensorDataset(torch.Tensor(np.array(val_fing)).squeeze(1), torch.Tensor(np.array(val_super).squeeze(1)))    
        test_dataset = TensorDataset(torch.Tensor(np.array(test_fing)).squeeze(1), torch.Tensor(np.array(test_super).squeeze(1)))
    elif TARGET_TYPE == 'class':
        train_dataset = TensorDataset(torch.Tensor(np.array(train_finger)).squeeze(1), torch.Tensor(np.array(train_class).squeeze(1))) # n_BATCH x n_FEATURES
        val_dataset = TensorDataset(torch.Tensor(np.array(val_fing)).squeeze(1), torch.Tensor(np.array(val_class).squeeze(1)))    
        test_dataset = TensorDataset(torch.Tensor(np.array(test_fing)).squeeze(1), torch.Tensor(np.array(test_class).squeeze(1)))
    else:
        raise ValueError("TARGET_TYPE must be 'pathway', 'superclass' or 'class'")
    
    # Create DataLoader objects
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
