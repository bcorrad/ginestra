import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 
from utils.utils import select_class_idx
import torch
from config import MODEL
from fingerprint_handler import calculate_fingerprint
import os

from torch_geometric.loader import DataLoader
from utils.graph_data_def import create_pytorch_geometric_graph_data_list_from_smiles_and_labels

np.random.seed(42)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

from config import DATADIR, N_SAMPLES, BATCH_SIZE, RANDOMIZE_SAMPLES, CLS_LIST, TARGET_MODE, TARGET_TYPE

with open(f'{DATADIR}/char2idx_class_V1.pkl','rb') as f:
    class_  = pickle.load(f)
with open(f'{DATADIR}/char2idx_super_V1.pkl','rb') as f:
    superclass_  = pickle.load(f)
with open(f'{DATADIR}/char2idx_path_V1.pkl','rb') as f:
    pathway_  = pickle.load(f)
with open(f'{DATADIR}/datset_class_all_V1.pkl','rb') as r:
    dataset = pickle.load(r)
    
from typing import Union, Literal
def data_generation(idx, data, 
                    spec_mol_smile=None, 
                    save_dataset:Union[Literal["train"], Literal["val"], Literal["test"]]=None):
    # Generate docstring
    """
    Generate data for training, validation, and test set
    Parameters
    ----------
    idx : list
        List of indices of the molecules to be selected
    data : dict
        Dictionary of the dataset
    spec_mol_smile : str, optional
        Specific molecule to be excluded from the dataset, by default None
    save_dataset : Union[Literal["train"], Literal["val"], Literal["test"]], optional
        Save the dataset to a file, by default None
    Returns
    -------
    list
        List of SMILES
    list
        List of fingerprints
    list
        List of pathways
    list
        List of superclass
    list
        List of class
    """

    smiles_list = []
    Y_train_path = []
    Y_train_super = []
    Y_train_class = []
    m_fingerprint_list = []

    for i, n in enumerate(idx):
        smiles = data[n]['SMILES']
        # if MODEL.lower() == "mlp":
        m_fingerprint_list.append(np.concatenate(calculate_fingerprint(smiles, 2), axis=1))
        if spec_mol_smile is not None:
            if smiles != spec_mol_smile:
                pass
        smiles_list.append(smiles)
        Y_train_path.append(data[n]['Pathway'])
        Y_train_super.append(data[n]['Super_class'])
        Y_train_class.append(data[n]['Class'])

    if save_dataset:
        dictionary = {'smiles': smiles_list,
                    'fingerprint': m_fingerprint_list,
                    'y_path': Y_train_path,
                    'y_super': Y_train_super,
                    'y_class': Y_train_class} 
        with open(f'{DATADIR}/{save_dataset}_dataset.pkl', 'wb') as f:
            pickle.dump(dictionary, f)
        print(f"{save_dataset} dataset saved successfully.")
    return smiles_list, m_fingerprint_list, Y_train_path, Y_train_super, Y_train_class 

# Train, Validation, and test set 
molecule_inchikey = list(dataset.keys())
np.random.shuffle(molecule_inchikey)
dict_ = np.array(molecule_inchikey)
Y_ = np.array([np.max(np.where(dataset[i]['Class']==1)[0]) for i in dict_]) # The maximum index of the class array where the value is 1

train_D, test_dict, y_train, y_test = train_test_split(dict_, Y_, test_size=0.2, random_state=1, stratify=Y_)
train_dict, val_dict, y_train, y_val = train_test_split(train_D, y_train, test_size=0.2, random_state=1, stratify=y_train)

if N_SAMPLES is None:

    if MODEL.lower() == "gin" or MODEL.lower() == "gine":
    # Load Dataloaders or Dataset
        dataloader_paths = [
            f'{DATADIR}/train_dataloader_gine.pkl',
            f'{DATADIR}/val_dataloader_gine.pkl',
            f'{DATADIR}/test_dataloader_gine.pkl'
        ]
        
        if all(os.path.exists(path) for path in dataloader_paths):
            print("Loading dataloaders...")
            train_dataloader = load_pickle(dataloader_paths[0])
            val_dataloader = load_pickle(dataloader_paths[1])
            test_dataloader = load_pickle(dataloader_paths[2])
            print("Dataloaders loaded successfully.")
        else:   
            print("Dataloader files not found. Please generate datasets first.")
            exit()
    else:
        dataset_paths = {
            'train': f'{DATADIR}/train_dataset.pkl',
            'val': f'{DATADIR}/val_dataset.pkl',
            'test': f'{DATADIR}/test_dataset.pkl'
        }
        
        datasets = {}
        for key, path in dataset_paths.items():
            if os.path.exists(path):
                print(f"Loading {key} dataset...")
                datasets[key] = load_pickle(path)
                print(f"{key.capitalize()} dataset loaded successfully.")
            else:
                print(f"{key.capitalize()} dataset file not found.")
                exit()

else:

    if not os.path.exists(f'{DATADIR}/train_dataset_w_fingerprints.pkl') \
        or not os.path.exists(f'{DATADIR}/val_dataset_w_fingerprints.pkl') \
            or not os.path.exists(f'{DATADIR}/test_dataset_w_fingerprints.pkl'):
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

    elif os.path.exists(f'{DATADIR}/train_dataset_w_fingerprints.pkl') \
        and os.path.exists(f'{DATADIR}/val_dataset_w_fingerprints.pkl') \
            and os.path.exists(f'{DATADIR}/test_dataset_w_fingerprints.pkl'):
        
        print("Loading datasets...")
        with open(f'{DATADIR}/train_dataset_w_fingerprints.pkl', 'rb') as f:
            train_dict = pickle.load(f)
        with open(f'{DATADIR}/val_dataset_w_fingerprints.pkl', 'rb') as f:
            val_dict = pickle.load(f)
        with open(f'{DATADIR}/test_dataset_w_fingerprints.pkl', 'rb') as f:
            test_dict = pickle.load(f)
        print("Datasets loaded successfully.")

        train_smiles, train_fing, train_path, train_super, train_class = train_dict['smiles'], train_dict['fingerprint'], train_dict['y_path'], train_dict['y_super'], train_dict['y_class']
        val_smiles, val_fing, val_path, val_super, val_class = val_dict['smiles'], val_dict['fingerprint'], val_dict['y_path'], val_dict['y_super'], val_dict['y_class']
        test_smiles, test_fing, test_path, test_super, test_class = test_dict['smiles'], test_dict['fingerprint'], test_dict['y_path'], test_dict['y_super'], test_dict['y_class']

    # Selezione delle classi in base al TARGET_TYPE    
    if TARGET_TYPE == "pathway":
        print("Training")
        train_paths, train_smiles_pathway, train_finger = select_class_idx(train_path, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
        print("Validation")
        val_paths, val_smiles_pathway, val_fing = select_class_idx(val_path, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
        print("Test")
        test_paths, test_smiles_pathway, test_fing = select_class_idx(test_path, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
    elif TARGET_TYPE == "superclass":
        train_super, train_smiles_super, train_finger = select_class_idx(train_super, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
        val_super, val_smiles_super, val_fing = select_class_idx(val_super, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
        test_super, test_smiles_super, test_fing = select_class_idx(test_super, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
    elif TARGET_TYPE == "class":
        train_class, train_smiles_class, train_finger = select_class_idx(train_class, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
        val_class, val_smiles_class, val_fing = select_class_idx(val_class, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
        test_class, test_smiles_class, test_fing  = select_class_idx(test_class, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)

    # Creazione dei DataLoader
    if MODEL.lower() != "mlp":
        train_df = pd.DataFrame({
            'SMILES': train_smiles_pathway if TARGET_TYPE == "pathway" else train_smiles_super
            if TARGET_TYPE == "superclass" else train_smiles_class,
            'Labels': train_paths.tolist() if TARGET_TYPE == "pathway" else train_super.tolist()
            if TARGET_TYPE == "superclass" else train_class,
            'Fingerprint': train_finger.squeeze(1).tolist()  # n_BATCH x n_FEATURES
        })

        val_df = pd.DataFrame({
            'SMILES': val_smiles_pathway if TARGET_TYPE == "pathway" else val_smiles_super
            if TARGET_TYPE == "superclass" else val_smiles_class,
            'Labels': val_paths.tolist() if TARGET_TYPE == "pathway" else val_super.tolist()
            if TARGET_TYPE == "superclass" else val_class,
            'Fingerprint': val_fing.squeeze(1).tolist()
        })

        test_df = pd.DataFrame({
            'SMILES': test_smiles_pathway if TARGET_TYPE == "pathway" else test_smiles_super
            if TARGET_TYPE == "superclass" else test_smiles_class,
            'Labels': test_paths.tolist() if TARGET_TYPE == "pathway" else test_super.tolist()
            if TARGET_TYPE == "superclass" else test_class,
            'Fingerprint': test_fing.squeeze(1).tolist()
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
        train_dataset = TensorDataset(torch.Tensor(np.array(train_finger)).squeeze(1), torch.Tensor(np.array(train_paths)))
        val_dataset = TensorDataset(torch.Tensor(np.array(val_fing)).squeeze(1), torch.Tensor(np.array(val_paths)))    
        test_dataset = TensorDataset(torch.Tensor(np.array(test_fing)).squeeze(1), torch.Tensor(np.array(test_paths)))  

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)

    # # Salvataggio dei dataloader generati  (Non serve)
    # if N_SAMPLES is None:
    #     save_pickle(train_dataloader, f'{DATADIR}/train_dataloader_{MODEL.lower()}.pkl')
    #     save_pickle(val_dataloader, f'{DATADIR}/val_dataloader_{MODEL.lower()}.pkl')
    #     save_pickle(test_dataloader, f'{DATADIR}/test_dataloader_{MODEL.lower()}.pkl')
    #     print("Dataloaders saved successfully.")

    print("DataLoader created successfully.")



# if N_SAMPLES is None:
#     #Load the dataloaders
#     if os.path.exists(f'{DATADIR}/train_dataloader_{MODEL.lower()}.pkl') and os.path.exists(f'{DATADIR}/val_dataloader_{MODEL.lower()}.pkl') and os.path.exists(f'{DATADIR}/test_dataloader_{MODEL.lower()}.pkl'):
#         print("Loading dataloaders...")
#         with open(f'{DATADIR}/train_dataloader_{MODEL.lower()}.pkl', 'rb') as f:
#             train_dataloader = pickle.load(f)
#         with open(f'{DATADIR}/val_dataloader_{MODEL.lower()}.pkl', 'rb') as f:
#             val_dataloader = pickle.load(f)
#         with open(f'{DATADIR}/test_dataloader_{MODEL.lower()}.pkl', 'rb') as f:
#             test_dataloader = pickle.load(f)
#         print("Dataloaders loaded successfully.")

#     if os.path.exists(f'{DATADIR}/train_dataset.pkl'):
#         print("Loading training dataset...")
#         with open(f'{DATADIR}/train_dataset.pkl', 'rb') as f:
#             train_dict = pickle.load(f)
#         train_smiles, train_fing, train_path, train_super, train_class = train_dict['smiles'], train_dict['fingerprint'], train_dict['y_path'], train_dict['y_super'], train_dict['y_class']
#         print("Training dataset loaded successfully.")

# else: 
#     train_smiles, train_fing, train_path, train_super, train_class = data_generation(train_dict, dataset, save_dataset="train")

#     if os.path.exists(f'{DATADIR}/val_dataset.pkl'):
#         print("Loading validation dataset...")
#         with open(f'{DATADIR}/val_dataset.pkl', 'rb') as f:
#             val_dict = pickle.load(f)
#         val_smiles, val_fing, val_path, val_super, val_class = val_dict['smiles'], val_dict['fingerprint'], val_dict['y_path'], val_dict['y_super'], val_dict['y_class']
#         print("Validation dataset loaded successfully.")
#     else:
#         val_smiles, val_fing, val_path, val_super, val_class = data_generation(val_dict, dataset, save_dataset="val")

#     if os.path.exists(f'{DATADIR}/test_dataset.pkl'):
#         print("Loading test dataset...")
#         with open(f'{DATADIR}/test_dataset.pkl', 'rb') as f:
#             test_dict = pickle.load(f)
#         test_smiles, test_fing, test_path, test_super, test_class = test_dict['smiles'], test_dict['fingerprint'], test_dict['y_path'], test_dict['y_super'], test_dict['y_class']
#         print("Test dataset loaded successfully.")
#     else:
#         test_smiles, test_fing, test_path, test_super, test_class  = data_generation(test_dict, dataset, save_dataset="test")     

#     # if CLS_LIST is None and TARGET_TYPE == "pathway":
#     #     CLS_LIST = list(range(len(train_path[0])))
#     # elif CLS_LIST is None and TARGET_TYPE == "superclass":
#     #     CLS_LIST = list(range(len(train_super[0])))
#     # elif CLS_LIST is None and TARGET_TYPE == "class":
#     #     CLS_LIST = list(range(len(train_class[0])))

#     if TARGET_TYPE == "pathway":
#         print("Training")
#         train_paths, train_smiles_pathway, train_finger = select_class_idx(train_path, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
#         print("Validation")
#         val_paths, val_smiles_pathway, val_fing = select_class_idx(val_path, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
#         print("Test")
#         test_paths, test_smiles_pathway, test_fing = select_class_idx(test_path, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
#     elif TARGET_TYPE == "superclass":
#         train_super, train_smiles_super, train_finger = select_class_idx(train_super, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
#         val_super, val_smiles_super, val_fing = select_class_idx(val_super, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
#         test_super, test_smiles_super, test_fing = select_class_idx(test_super, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
#     elif TARGET_TYPE == "class":
#         train_class, train_smiles_class, train_finger = select_class_idx(train_class, train_smiles, train_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
#         val_class, val_smiles_class, val_fing = select_class_idx(val_class, val_smiles, val_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)
#         test_class, test_smiles_class, test_fing  = select_class_idx(test_class, test_smiles, test_fing, CLS_LIST, TARGET_MODE, n_samples=N_SAMPLES, shuffle=RANDOMIZE_SAMPLES)

#     if MODEL.lower() != "mlp":
#         train_df = pd.DataFrame({
#             'SMILES': train_smiles_pathway if TARGET_TYPE == "pathway" else train_smiles_super
#             if TARGET_TYPE == "superclass" else train_smiles_class,
#             'Labels': train_paths.tolist() if TARGET_TYPE == "pathway" else train_super.tolist()
#             if TARGET_TYPE == "superclass" else train_class
#         })

#         val_df = pd.DataFrame({
#             'SMILES': val_smiles_pathway if TARGET_TYPE == "pathway" else val_smiles_super
#             if TARGET_TYPE == "superclass" else val_smiles_class,
#             'Labels': val_paths.tolist() if TARGET_TYPE == "pathway" else val_super.tolist()
#             if TARGET_TYPE == "superclass" else val_class
#         })

#         test_df = pd.DataFrame({
#             'SMILES': test_smiles_pathway if TARGET_TYPE == "pathway" else test_smiles_super
#             if TARGET_TYPE == "superclass" else test_smiles_class,
#             'Labels': test_paths.tolist() if TARGET_TYPE == "pathway" else test_super.tolist()
#             if TARGET_TYPE == "superclass" else test_class
#         })

#         from torch_geometric.loader import DataLoader
#         from utils.graph_data_def import create_pytorch_geometric_graph_data_list_from_smiles_and_labels
#         train_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(train_df)
#         val_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(val_df)
#         test_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(test_df)

#         train_dataloader = DataLoader(train_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
#         val_dataloader = DataLoader(val_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
#         test_dataloader = DataLoader(test_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
#     else:
#         from torch.utils.data import DataLoader, TensorDataset
#         train_dataset = TensorDataset(torch.Tensor(np.array(train_finger)).squeeze(1), torch.Tensor(np.array(train_paths)))
#         val_dataset = TensorDataset(torch.Tensor(np.array(val_fing)).squeeze(1), torch.Tensor(np.array(val_paths)))    
#         test_dataset = TensorDataset(torch.Tensor(np.array(test_fing)).squeeze(1), torch.Tensor(np.array(test_paths)))  

#         train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
#         val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
#         test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)

#     # Save the dataloaders
#     with open(f'{DATADIR}/train_dataloader_{MODEL.lower()}.pkl', 'wb') as f:
#         pickle.dump(train_dataloader, f)
#     with open(f'{DATADIR}/val_dataloader_{MODEL.lower()}.pkl', 'wb') as f:
#         pickle.dump(val_dataloader, f)
#     with open(f'{DATADIR}/test_dataloader_{MODEL.lower()}.pkl', 'wb') as f:
#         pickle.dump(test_dataloader, f)

#     print("DataLoader created successfully.")




