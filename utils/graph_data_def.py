'''Pytorch Geometric  ---> Graph object generation, Multilabel Classification Task: y=[7], y_superclass=[70], y_class=[653]'''

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from typing import List, Union, Literal

def encode(x, permitted_list: List=None, encoding: Union[Literal['hot'], Literal['label']]='hot'):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if permitted_list is None:
        raise ValueError("permitted_list cannot be None") 

    if x not in permitted_list:
        x = permitted_list[-1]

    if "hot" in encoding.lower():
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

        return binary_encoding
    
    elif "label" in encoding.lower():
        return [permitted_list.index(x)+1]

def get_atom_features(atom, 
                      use_chirality = False, 
                      hydrogens_implicit = False):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    permitted_list_of_atoms = ['S', 'Sn', 'In', 'Br', 'F', 'Cl', 'B', 'N', 'O', 'I', 'C', 'Co', 'P'] #TODO: aggiungere Unkwown
    permitted_list_of_atoms_old =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    atom_type_enc = encode(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = encode(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    # formal_charge_enc = encode(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    formal_charge_enc = [int(atom.GetFormalCharge())]
    hybridisation_type_enc = encode(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc
    # print('Atom features vector ',len(atom_feature_vector))
    # + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = encode(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = encode(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)
    # return a random vector with the same shape as atom_feature_vector. 
    # return np.array([np.random.rand() for _ in range(len(atom_feature_vector))])

def get_bond_features(bond, 
                      use_stereochemistry=True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = encode(bond.GetBondType(), permitted_list=permitted_list_of_bond_types, encoding="hot")  # shape 4, encoding="label"
    bond_is_conj_enc = [int(bond.GetIsConjugated())]                                    # shape 1
    bond_is_in_ring_enc = [int(bond.IsInRing())]                                        # shape 1
    # TODO: ATTENZIONE QUI
    bond_feature_vector = \
        bond_type_enc + \
        bond_is_conj_enc + \
        bond_is_in_ring_enc        # shape 4 + 1 + 1 = 6   
    
    if use_stereochemistry == True:
        stereo_type_enc = encode(str(bond.GetStereo()), permitted_list=["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"], encoding="hot")
        bond_feature_vector += stereo_type_enc  # if one hot encoding: shape 6 + 4 = 10
    
    return np.array(bond_feature_vector)
    # Return a random vector with the same shape as bond_feature_vector.
    # return np.array([np.random.rand() for _ in range(len(bond_feature_vector))])


def convert_pathway_labels(_labels):
    """
    Converte una lista di stringhe in tensori PyTorch
    """
    labels_np = np.array(np.fromstring(label.strip("[]"), sep=" ") for label in _labels)
    labels_tensor = torch.Tensor(labels_np)
    return labels_tensor


from typing import Union, Literal
def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(df, mode:Union[Literal['edge_adjacency'], None]=None):
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
        y = molecule['Labels']
        fingerprint = molecule['Fingerprint']
        
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











