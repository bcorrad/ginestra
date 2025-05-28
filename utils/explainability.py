from rdkit import Chem
from rdkit import Chem

def pyg_data_to_mol(data, permitted_list_of_atoms):
    mol = Chem.RWMol()

    # Aggiungi atomi
    for i in range(data.num_nodes):
        idx = int(data.x[i][0].item())  # ID intero dell’atomo
        if idx >= len(permitted_list_of_atoms):
            raise ValueError(f"Atom index {idx} out of bounds for permitted list.")
        symbol = permitted_list_of_atoms[idx]
        mol.AddAtom(Chem.Atom(symbol))

    # Aggiungi legami
    for k in range(data.edge_index.size(1)):
        i = int(data.edge_index[0, k].item())
        j = int(data.edge_index[1, k].item())

        # Evita doppie aggiunte se bidirezionale
        if i < j:
            bond_type = Chem.BondType.SINGLE  # default
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                bond_val = int(data.edge_attr[k][0].item())
                bond_type = {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                    4: Chem.BondType.AROMATIC
                }.get(bond_val, Chem.BondType.SINGLE)
            mol.AddBond(i, j, bond_type)

    # Finalizzazione
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def draw_molecule_with_attention(
    mol,
    edge_index,
    attn_weights,
    threshold=0.1,
    save_path="mol_with_colorbar_fixed.png",
    dpi=300,
    cmap_name="viridis"
):
    # Costruzione dizionario attenzione
    attn_dict = {}
    for (i, j), attn in zip(edge_index.t().tolist(), attn_weights.tolist()):
        key = tuple(sorted((i, j)))
        attn_dict[key] = max(attn_dict.get(key, 0), attn)

    max_attn = max(attn_dict.values()) if attn_dict else 1.0
    min_attn = min(attn_dict.values()) if attn_dict else 0.0
    cmap = cm.get_cmap(cmap_name)

    # Costruzione colori legami
    bond_colors = {}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        key = tuple(sorted((a1, a2)))
        if key in attn_dict and attn_dict[key] > threshold:
            norm_attn = (attn_dict[key] - min_attn) / (max_attn - min_attn + 1e-8)
            rgba = cmap(norm_attn)
            bond_colors[bond.GetIdx()] = rgba

    # Disegna molecola
    mol = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(
        mol,
        highlightAtoms=[],
        highlightBonds=list(bond_colors.keys()),
        highlightAtomColors={},
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # 🎯 Colorbar sottile con etichette leggibili
    fig, ax = plt.subplots(figsize=(0.35, 4), dpi=dpi)
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(smap, cax=ax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=7, length=3, direction='in', pad=2)
    cb.set_label("Attention", fontsize=8, labelpad=5)

    # Forza layout senza tight
    fig.subplots_adjust(left=0.2, right=0.8)

    fig.canvas.draw()
    colorbar_img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)

    # Resize altezza
    colorbar_img = colorbar_img.resize((colorbar_img.width, mol_img.height))

    # Combina
    total_width = mol_img.width + colorbar_img.width
    result = Image.new("RGB", (total_width, mol_img.height), color=(255, 255, 255))
    result.paste(mol_img, (0, 0))
    result.paste(colorbar_img, (mol_img.width, 0))

    result.save(save_path, dpi=(dpi, dpi))
    print(f"✅ Molecola con colorbar visibile salvata in: {save_path}")

import torch 
from torch_geometric.data import Data

attn_h2_ = attn_h3[-1].mean(dim=1) if attn_h3[-1].dim() == 2 else attn_h3
data = Data(x=x, edge_index=edge_index)
permitted_list_of_atoms = ['S', 'Sn', 'In', 'Br', 'F', 'Cl', 'B', 'N', 'O', 'I', 'C', 'Co', 'P'] 
mol = pyg_data_to_mol(data, permitted_list_of_atoms)
draw_molecule_with_attention(mol, edge_index, attn_h2_, threshold=0.1)
