from config import DATADIR, TARGET_TYPE, N_SAMPLES, USE_AVAILABLE_DATASET, MODELS, EXPERIMENT_FOLDER
from typing import Union, Literal
import pickle
import os
from fingerprint_handler import calculate_fingerprint
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from utils.graph_data_def import get_atom_features, get_bond_features
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from config import BATCH_SIZE


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.feature_dim = len(df.iloc[0]['fingerprint'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles = self.df.iloc[idx]['SMILES']
        fingerprint = self.df.iloc[idx]['fingerprint']
        label = self.df.iloc[idx][TARGET_TYPE.capitalize()]
        return smiles, fingerprint, label
    

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


def dataset_split(matrix, n_samples=N_SAMPLES):
    """
    Returns the unique vectors (rows) of a given matrix.

    Parameters:
        matrix (np.ndarray): A 2D numpy array.
        n_samples (int, optional): The number of samples to select from each class. Defaults to config.N_SAMPLES (all the samples if None).

    Returns:
        np.ndarray: A 2D numpy array containing the unique rows of the input matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if isinstance(matrix, list):
        print("Input is a list, converting to numpy array.")
        matrix = np.array(matrix)
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")
    
    # Search the multi-label samples (if any row has more than one 1)
    sum_rows = np.sum(matrix, axis=1)
    single_label_rows = np.where(sum_rows == 1)[0]
    matrix_single_label = matrix[single_label_rows]
    
    # Find the unique rows
    unique_rows = np.unique(matrix_single_label, axis=0)
    
    # Count the number of occurrences of each unique row
    unique_counts = np.array([np.sum(np.all(matrix_single_label == row, axis=1)) for row in unique_rows])
    
    # Save the indices of matrix containing the unique rows (eg, a list of indices of all the samples of class i)
    classwise_indices = []
    
    # Find the indices of each unique row in the original matrix
    for i, row in enumerate(unique_rows):
        indices = np.where(np.all(matrix_single_label == row, axis=1))[0]
        if len(indices) > 1:
            classwise_indices.append(indices)
            print(f"Row {i} corresponding to class {np.argmax(row)} occurs {len(indices)} times in the original matrix.")
    
    # Equally distribute the samples of each class using the classwise indices
    training_indices = []
    validation_indices = []
    test_indices = []
    
    training_samples = []
    validation_samples = []
    test_samples = []
    
    # Split the indices into training, validation, and test sets
    training_split = 0.6
    validation_split = 0.2
    # test_split = 1 - training_split - validation_split
    
    for indices in classwise_indices:
        np.random.shuffle(indices)
        n = len(indices)
        # Update the number of samples if N_SAMPLES is not None
        if n_samples is not None:
            if len(indices) > n_samples:
                print(f"Class {np.argmax(matrix_single_label[indices[0]])} has more than {n_samples} samples. Randomly selecting {n_samples} samples.")
                indices = indices[:n_samples]
                n = n_samples
            else:
                n = len(indices)
                
        train_end = int(training_split * n)
        val_end = int((training_split + validation_split) * n)
        training_indices.extend(indices[:train_end])
        validation_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])
        training_samples.extend(matrix_single_label[indices[:train_end]])
        validation_samples.extend(matrix_single_label[indices[train_end:val_end]])
        test_samples.extend(matrix_single_label[indices[val_end:]])
        print(f"Class {np.argmax(matrix_single_label[indices[0]])} - Training: {len(training_indices)}, Validation: {len(validation_indices)}, Test: {len(test_indices)}")

    training_samples = np.array(training_samples)
    validation_samples = np.array(validation_samples)
    test_samples = np.array(test_samples)
    
    return training_indices, validation_indices, test_indices, training_samples, validation_samples, test_samples


def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(df, 
                                                                    mode:Union[Literal['edge_adjacency'], None]=None,
                                                                    target:Union[Literal['Pathway'], Literal['Superclass'], Literal['Class']]=None):
    """
    Inputs:
    df: DataFrame con colonne 'SMILES', 'Pathway', 'Super_class', 'Class', 'Fingerprint'.
    
    Outputs:
    data_list: lista di torch_geometric.data.Data objects che rappresentano grafi molecolari etichettati.
    """
    data_list = []
    
    # Itera sul dataframe
    from tqdm import tqdm
    import torch.nn.functional as F
    for _, molecule in tqdm(df.iterrows()):
        smiles = molecule['SMILES']
        y = molecule[target]
        try:
            fingerprint = molecule['Fingerprint']
        except KeyError:
            fingerprint = molecule['fingerprint']
        
        # Convertire SMILES in oggetto RDKit mol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Molecola non valida per SMILES {smiles}. Ignorata.")
            continue

        # Ottenere dimensioni delle feature
        n_nodes = mol.GetNumAtoms()
        # n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

        # Costruire la matrice di feature dei nodi X
        X_block_A = np.zeros((n_nodes, n_node_features))
        for atom_idx, atom in enumerate(mol.GetAtoms()):
            X_block_A[atom.GetIdx(), :] = get_atom_features(atom)

        if mode == 'edge_adjacency':
            X_block_B = torch.Tensor(np.stack([[[0] * len(y)] * n_nodes])[0])
            X_block_C = torch.Tensor(np.stack([[[0] * n_node_features] * len(y)])[0])
            # TODO: Fix
            X_block_D = np.stack([y] * len(y))
            # Concatenare i blocchi A e B
            X_block_AB = torch.cat((torch.Tensor(X_block_A), X_block_B), dim=1)
            # Concatenare i blocchi C e D
            X_block_CD = torch.cat((X_block_C, torch.Tensor(X_block_D)), dim=1)
            # Concatenare i blocchi AB e CD
            X = torch.Tensor(torch.cat((X_block_AB, X_block_CD), dim=0))
        else:
            X = torch.Tensor(X_block_A)

        # Costruire la matrice di indici degli archi E
        adj_matrix = GetAdjacencyMatrix(mol)                # boolean num_nodes x num_nodes
        
        # Costruire la matrice di indici degli archi E
        # E = torch.zeros((len(rows) + len(y), len(cols) + len(y)))  # num_edges x num_edges
        # We need to create a block matrix E = [A B; C D] where A is the adjacency matrix of the graph, 
        # B is a matrix of zeros, C is the transpose of B, and D is a matrix of zeros.
        ADJ_block_A = torch.Tensor(adj_matrix)
        if mode == 'edge_adjacency':
            ADJ_block_B = torch.Tensor(np.stack([[y] * adj_matrix.shape[0]])[0])
            ADJ_block_C = ADJ_block_B.T
            ADJ_block_D = torch.zeros((len(y), len(y)))
            # Concatenare i blocchi A e B
            ADJ_block_AB = torch.cat((ADJ_block_A, ADJ_block_B), dim=1)
            # Concatenare i blocchi C e D
            ADJ_block_CD = torch.cat((ADJ_block_C, ADJ_block_D), dim=1)
            # Concatenare i blocchi AB e CD
            ADJ = torch.cat((ADJ_block_AB, ADJ_block_CD), dim=0)
        else:
            ADJ = ADJ_block_A

        # Convertire la matrice di adiacenza in tensori
        # torch.Tensor of shape 2 x num_edges, where num_edges = 2 * num_bonds. 
        E = np.nonzero(ADJ).to(torch.long)
        E = torch.Tensor(E).T               # Edge index matrix with shape num_edges x 2 

        # Costruire l'array delle feature degli archi EF
        # Unpack the rows and columns of the adjacency matrix
        rows, cols = np.nonzero(adj_matrix)
        EF = np.zeros((len(rows), n_edge_features))
        for k, (i, j) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
        EF = torch.Tensor(EF)
        # EF = EF.T                           # Edge feature matrix with shape (num_edges x num_edge_features)

        # Convertire le etichette in tensori
        if isinstance(y, int):
            y_tensor = torch.Tensor([y])
        # if y is a list of integers, convert to tensor
        elif isinstance(y, np.ndarray) and y.ndim == 1:
            y_tensor = torch.Tensor(y).unsqueeze(0)
        elif isinstance(y, np.dtype) or isinstance(y, list):
            y_tensor = torch.Tensor(y).unsqueeze(0)

        # Creare l'oggetto Data
        # data = Data(
        #     x=X,
        #     edge_index=E,
        #     edge_attr=EF,
        #     y=y_tensor,
        #     fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float)
        # )  
        # data_list.append(data)

        data_args = {
        "x": X,
        "edge_index": E,
        "edge_attr": EF,
        "y": y_tensor}
    
        from config import USE_FINGERPRINT
        # Se USE_FINGERPRINT è True, aggiungi fingerprint
        if USE_FINGERPRINT:
            fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float)
            data_args["fingerprint_tensor"] = fingerprint_tensor

        # Creare l'oggetto Data dinamicamente
        data = Data(**data_args)
        data_list.append(data)

    return data_list


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


if N_SAMPLES is not None:
    suffix = f'_{N_SAMPLES}'
else:
    suffix = ''
    
gnn_train_dataloader = None
gnn_val_dataloader = None
gnn_test_dataloader = None
mlp_train_dataloader = None
mlp_val_dataloader = None
mlp_test_dataloader = None

if USE_AVAILABLE_DATASET:   
    if "gin" in MODELS or "gine" in MODELS:
        if (os.path.exists(f'{DATADIR}/train_geodataloader_{TARGET_TYPE}{suffix}.pkl') and \
            os.path.exists(f'{DATADIR}/val_geodataloader_{TARGET_TYPE}{suffix}.pkl') and \
                os.path.exists(f'{DATADIR}/test_geodataloader_{TARGET_TYPE}{suffix}.pkl')):
                    
            print("Dataset already exists for GIN/GINE. Loading from pickle files.")
            gnn_train_dataloader = load_pickle(f'{DATADIR}/train_geodataloader_{TARGET_TYPE}{suffix}.pkl')
            gnn_val_dataloader = load_pickle(f'{DATADIR}/val_geodataloader_{TARGET_TYPE}{suffix}.pkl')
            gnn_test_dataloader = load_pickle(f'{DATADIR}/test_geodataloader_{TARGET_TYPE}{suffix}.pkl')
    elif "mlp" in MODELS:
        if (os.path.exists(f'{DATADIR}/train_dataloader_{TARGET_TYPE}{suffix}.pkl') and \
            os.path.exists(f'{DATADIR}/val_dataloader_{TARGET_TYPE}{suffix}.pkl') and \
                os.path.exists(f'{DATADIR}/test_dataloader_{TARGET_TYPE}{suffix}.pkl')):
                    
            print("Dataset already exists for MLP. Loading from pickle files.")
            mlp_train_dataloader = load_pickle(f'{DATADIR}/train_dataloader_{TARGET_TYPE}{suffix}.pkl')
            mlp_val_dataloader = load_pickle(f'{DATADIR}/val_dataloader_{TARGET_TYPE}{suffix}.pkl')
            mlp_test_dataloader = load_pickle(f'{DATADIR}/test_dataloader_{TARGET_TYPE}{suffix}.pkl')
else:
    print("Dataset does not exist. Generating new dataset.")
    # Load data from pkl files
    with open(f'{DATADIR}/char2idx_class_V1.pkl','rb') as f:
        class_  = pickle.load(f)
    with open(f'{DATADIR}/char2idx_super_V1.pkl','rb') as f:
        superclass_  = pickle.load(f)
    with open(f'{DATADIR}/char2idx_path_V1.pkl','rb') as f:
        pathway_  = pickle.load(f)
    with open(f'{DATADIR}/datset_class_all_V1.pkl','rb') as r:
        dataset = pickle.load(r)
        # Remove _ from the keys of the dictionary at the second level (Super_class -> Superclass)
        dataset = {k: {k2.replace("_", ""): v2 for k2, v2 in v.items()} for k, v in dataset.items()}

    # Train, Validation, and test set 
    molecule_inchikey = list(dataset.keys())
    np.random.shuffle(molecule_inchikey)
    molecule_dict = np.array(molecule_inchikey)

    # MULTICLASS MODE: Drop the dataset[i] elements that sum to > 1 for the key TARGET_TYPE
    dataset = {k: v for k, v in dataset.items() if np.sum(v[TARGET_TYPE.capitalize()]) == 1}

    # Build the list of SMILES
    smiles_df = []
    for i in dataset.keys():
        smiles_df.append(dataset[i]['SMILES'])
    smiles_df = np.array(smiles_df)

    # Build the list of fingerprints
    fingerprint_list = [] 
    for i in smiles_df:
        fingerprint_list.append(np.concatenate(calculate_fingerprint(i, 2), axis=1))

    # Build the list of targets (Class, Super_class, Pathway)
    labels_list = []
    for i in dataset.keys():
        labels_list.append(dataset[i][TARGET_TYPE.capitalize()])

    # Create a dataframe with the SMILES, fingerprints, and labels
    df = pd.DataFrame({'SMILES': smiles_df, 'fingerprint': fingerprint_list, TARGET_TYPE.capitalize(): labels_list})

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the dataframe into training, validation, and test sets using the dataset_split function
    train_indices, val_indices, test_indices, train_samples, val_samples, test_samples = dataset_split(np.array(labels_list))
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    test_df = df.iloc[test_indices]
    
    # Save the train_df, val_df, and test_df to pickle files
    if "mlp" in MODELS:
        # Convert train_df, val_df, and test_df to PyTorch Dataset objects
        train_dataset = CustomDataset(train_df)
        val_dataset = CustomDataset(val_df)
        test_dataset = CustomDataset(test_df)
        # Create DataLoader objects
        mlp_train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        mlp_val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        mlp_test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Save the DataLoader objects to pickle files
        save_pickle(mlp_train_dataloader, f'{DATADIR}/train_dataloader_{TARGET_TYPE}{suffix}.pkl')
        save_pickle(mlp_val_dataloader, f'{DATADIR}/val_dataloader_{TARGET_TYPE}{suffix}.pkl')
        save_pickle(mlp_test_dataloader, f'{DATADIR}/test_dataloader_{TARGET_TYPE}{suffix}.pkl')
    
    elif "gin" in MODELS or "gine" in MODELS:
        # Convert train_df, val_df, and test_df to PyTorch Geometric DataLoader objects
        train_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(train_df, target=TARGET_TYPE.capitalize())
        val_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(val_df,  target=TARGET_TYPE.capitalize())
        test_datalist = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(test_df,  target=TARGET_TYPE.capitalize())

        gnn_train_dataloader = GeoDataLoader(train_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
        gnn_val_dataloader = GeoDataLoader(val_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
        gnn_test_dataloader = GeoDataLoader(test_datalist, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)

        # Save the GeoDataLoader objects to pickle files
        save_pickle(gnn_train_dataloader, f'{DATADIR}/train_geodataloader_{TARGET_TYPE}{suffix}.pkl')
        save_pickle(gnn_val_dataloader, f'{DATADIR}/val_geodataloader_{TARGET_TYPE}{suffix}.pkl')
        save_pickle(gnn_test_dataloader, f'{DATADIR}/test_geodataloader_{TARGET_TYPE}{suffix}.pkl')