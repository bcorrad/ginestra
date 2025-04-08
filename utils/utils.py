import numpy as np
from config import MODEL, DATADIR
from typing import Union, Literal
import pickle
from fingerprint_handler import calculate_fingerprint

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


def select_class_idx_path(target_vectors: list, 
                     smiles: list, 
                     m_fingrprints: list=None,
                     class_indexes: list=None, 
                     label_enc:Union[Literal["onehot"], Literal["one-hot"]]="onehot",
                     n_samples:int=None, 
                     shuffle:bool=True
                     ):
    
    from config import LABELS_CODES
    datasplit_targets = []
    datasplit_smiles_classes = []
    datasplit_m_fingerprints = []
    #TODO: sistemare dopo che decidiamo se aumentare il numero di classi o no
    #num_classes = len(target_vectors[0])
    num_classes = len(class_indexes) if class_indexes is not None else len(LABELS_CODES.keys())
    
    for label_key, label_val in LABELS_CODES.items(): 
        # idx_tmp = np.argwhere(np.array(target_vectors)==label_val)[:, 0]
        idx_tmp = np.argwhere(np.array([str(v) for v in np.array(target_vectors)])==np.array([str(vv) for vv in label_val[None,:]]))  
        if n_samples is not None and n_samples <= len(idx_tmp):
            idx_ = idx_tmp[:n_samples]  # idx_.shape = (batch_size, 1)
            print(f"N_SAMPLES available for class {label_key}, {n_samples} examples taken.")
        else:
            print(f"N_SAMPLES not available for class {label_key}, {len(idx_tmp)} examples taken.")
            n_samples = len(idx_tmp)
            idx_ = idx_tmp
        
        idx_ = idx_.flatten()       # idx_.shape = (batch_size,)
        
        # datasplit_targets.append(np.array(target_vectors)[idx_])
        if "hot" in label_enc:
            datasplit_targets.append(np.array([np.eye(num_classes)[label_key]]*n_samples))
        else:
            datasplit_targets.append([label_key]*n_samples)

        datasplit_smiles_classes.append(np.array(smiles)[idx_])

        datasplit_m_fingerprints.append(np.array(m_fingrprints)[idx_])
        # if MODEL.lower() == "mlp" and m_fingrprints is not None:
        #     datasplit_m_fingerprints.append(np.array(m_fingrprints)[idx_])
        # else:
        #     datasplit_m_fingerprints = None
    
    # Randomize the order of the samples
    datasplit_targets = np.concatenate(datasplit_targets)
    datasplit_smiles_classes = np.concatenate(datasplit_smiles_classes)
    # if datasplit_m_fingerprints is not None and len(datasplit_m_fingerprints) > 0:
    #     datasplit_m_fingerprints = np.concatenate(datasplit_m_fingerprints)
    datasplit_m_fingerprints = np.concatenate(datasplit_m_fingerprints)
    indexes = np.arange(len(datasplit_smiles_classes)).astype(int)
    if shuffle:
        np.random.shuffle(indexes)
    if datasplit_m_fingerprints is not None:
        return datasplit_targets[indexes], datasplit_smiles_classes[indexes], datasplit_m_fingerprints[indexes]
    else:
        return datasplit_targets[indexes], datasplit_smiles_classes[indexes], None


# def select_class_idx(target_vectors: list, 
#                      smiles: list, 
#                      m_fingrprints: list=None,
#                      class_indexes: list=None, 
#                      label_enc:Union[Literal["onehot"], Literal["one-hot"]]="onehot",
#                      n_samples:int=None, 
#                      shuffle:bool=True
#                      ):
    
#     datasplit_targets = []
#     datasplit_smiles_classes = []
#     datasplit_m_fingerprints = []
#     #TODO: sistemare dopo che decidiamo se aumentare il numero di classi o no
#     num_classes = len(target_vectors[0])
#     #select indices of the single class in target vector

#     idx_single_class = np.argwhere(np.array(target_vectors)[np.argwhere(np.sum(target_vectors, axis=1)==1)[:,0]])[:,1]
#     idx_tmp = idx_single_class


#     if n_samples is not None and n_samples <= len(idx_tmp):
#         idx_ = idx_tmp[:n_samples]  # idx_.shape = (batch_size, 1)
#         print(f"N_SAMPLES available for class {idx_tmp}, {n_samples} examples taken.")
#     else:
#         print(f"N_SAMPLES not available for class {idx_tmp}, {len(idx_)} examples taken.")
#         n_samples = len(idx_)
    
#     datasplit_smiles_classes = np.array(smiles)[idx_single_class].tolist()

    


def select_class_idx_super(target_vectors: list, 
                     smiles: list, 
                     m_fingrprints: list=None,
                     class_indexes: list=None, 
                     label_enc:Union[Literal["onehot"], Literal["one-hot"]]="onehot",
                     n_samples:int=None, 
                     shuffle:bool=True
                    ):
    
    num_classes = len(target_vectors[0])
    # save Single class idx
    idx_single_class = np.argwhere(np.sum(target_vectors, axis=1)==1)
    
    # transformation in numpy array
    smiles, target_vectors = np.array(smiles), np.array(target_vectors)

    # save only values appartaining to Single calss 
    target_vectors = target_vectors[idx_single_class]
    smiles = smiles[idx_single_class]

    # apply argmax to recover the index (not one-hot encoded) target for the samples
    datasplit_path_single_class = np.argmax(target_vectors, axis=2)

    datasplit_targets = []
    datasplit_smiles_classes = []
    datasplit_m_fingerprints = []
    
    printed = False
    # Find unique values in target arrays
    unique_labels = np.unique(datasplit_path_single_class)
    # Iterate over unique labels and select samples
    for i, label in enumerate(unique_labels): 
        if printed == False:
            print(f"Class {i}: {label}")
            printed = True

        idx_tmp = np.argwhere(datasplit_path_single_class==label)[:, 0]  
        
        if n_samples is not None and n_samples <= len(idx_tmp):
            idx_ = idx_tmp[:n_samples]
        else:
            print(f"N_SAMPLES not available for class {label}, {len(idx_tmp)} examples taken.")
            idx_ = idx_tmp
        
        # datasplit_targets.append(datasplit_path_single_class[idx_])
        # datasplit_smiles_classes.append(smiles[idx_])
       
        idx_ = idx_.flatten() 

        datasplit_smiles_classes.append(np.array(smiles)[idx_])

        datasplit_m_fingerprints.append(np.array(m_fingrprints)[idx_])

        datasplit_targets.append(np.array(target_vectors)[idx_])
        #  labels = np.zeros((datasplit_targets.shape[0], label_translated.shape[0]))
        #  labels[np.argwhere(datasplit_targets == label)[:, 0]] = i

        printed = False
    # idx_shik = np.argwhere(datasplit_path_single_class==5)[:, 0]   #Class: Shikimates
    # datasplit_path_binary_shik = datasplit_path_single_class[idx_shik]
    # datasplit_smiles_binary_shik = smiles[idx_shik]
    datasplit_targets = np.concatenate(datasplit_targets, axis=0)
    datasplit_smiles_classes = np.concatenate(datasplit_smiles_classes, axis=0)
    datasplit_m_fingerprints = np.concatenate(datasplit_m_fingerprints, axis=0)
    # for i, label in enumerate(class_number):
    #     if label_type == "two_classes": #and len(class_number) == 2:
    #         datasplit_targets[np.argwhere(datasplit_targets == label)[:, 0]] = i    # 0 or 1
    #         paths_classes_to_return = datasplit_targets[:, 0]

    #     elif label_type == "ohe" or label_type == "binary":
    #         if i == 0:
    #             labels = np.zeros((len(datasplit_targets), num_classes))
    #         labels[np.argwhere(datasplit_targets == label)[:, 0], :] = np.eye(num_classes)[i]
    #         paths_classes_to_return = labels

    indexes = np.arange(len(datasplit_smiles_classes)).astype(int)
    if shuffle:
        np.random.shuffle(indexes)
    if datasplit_m_fingerprints is not None:
        return datasplit_targets[indexes], datasplit_smiles_classes[indexes], datasplit_m_fingerprints[indexes]
    else:
        return datasplit_targets[indexes], datasplit_smiles_classes[indexes], None