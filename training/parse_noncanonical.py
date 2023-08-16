import gzip
import numpy as np
import torch
import os,sys
import glob
import re
from scipy.spatial import KDTree
from itertools import combinations,permutations
import tempfile
import subprocess
from biopandas.pdb import PandasPdb
import pandas as pd
import argparse
from typing import List
from tqdm import tqdm

def extract_pdb_info(f:str,
                     to1letter: dict,
                     aa2idx: dict,
                     OUTPUT: str,
                     res_names: list,
                     ) -> List[str]:
    """Take in a pdb file and extract xyz, atom type, occupancy, and coordinates
    from the file.

    PARAMETERS
    ----------
    f : str
        pdb file path

    to1letter: dict
        convert 3 letter Amino Acid to 1 letter

    aa2idx: dict
        keys are (res, atom): idx_number

    OUTPUT: str
        global output path str

    res_names: list
        list of allowed residue types


    WRITTEN OUTPUTS
    ---------------
    FILE OUTPUT -> f_chid.pt

    seq: str
        AminoAcid Sequence

    xyz: torch.Tensor
        XYZ coordinates

    mask: torch.Tensor
        if an atom is present in the data or not
        1 for yes 0 for no

    occupancy: torch.Tensor
        0-1 value measuring disorder of an atom

    bfac: torch.Tensor
        For me this value is going to be zero for everything, as our methods
        are Ab-Initio in-silico. However, it is needed since the dataset has it.

    RETURNS
    -------
    seq_all: list
        list of One letter sequences of the peptides
    """
    # Check f is of type string
    assert isinstance(f, str),"This function was expecting a string path, and not a file object"
    assert os.path.exists(f),f"File {f} does not exist!"

    # Load data into pandas dataframe
    data = PandasPdb().read_pdb(f)
    filename = f.strip('.pdb').split('/')[-1]

    # Sequence holding object
    seq_all = []

    # Load in het and atom dfs
    het = data.df["HETATM"]
    atom = data.df["ATOM"]

    # Combine the dfs
    df_cat = pd.concat([het, atom], ignore_index=True)

    # Sort so the atoms are in order
    df_cat.sort_values(by="atom_number", inplace=True)

    # extra chain information
    chids = set(df_cat.chain_id)

    # loop through chids
    for id in chids:
        # extract out id
        temp = df_cat[df_cat.chain_id == id]

        # Get rid of waters
        # temp = temp[temp.residue_name != 'HOH']
        temp = temp[temp.residue_name.isin(res_names)]
        
        # extract sequence info by eliminating multiple residue number occurences
        temp_unique = temp.drop_duplicates(subset="residue_number",
                                           keep="first")

        # convert residue_name to a string
        seq = "".join(
            [to1letter["GLY"] if aaa not in to1letter.keys() else to1letter[aaa] for aaa in temp_unique.residue_name]
        )

        # If this chain has nothing but elements/unknown current noncanonicals then pass
        if seq == "":
            continue

        # extract seq length
        L = len(seq)
        ctr = -1

        # remove non useful atoms
        temp = temp[temp.atom_name.isin(
            ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"]
        )]

        # init xyz and occupancy
        xyz = torch.zeros((L,14,3)) #(L, Atom, XYZ)
        occupancy = torch.zeros((L,14)) #(L, Atom)
        mask = torch.zeros((L,14)) #(L, Atom)

        # hold val for checking position
        residue_idx = -1

        # fill in occupancy and xyz info
        for _, row in temp.iterrows():
            # extract residue number and this will be our (L) key for when to switch
            current_num = row.residue_number
            if current_num != residue_idx:
                residue_idx = current_num
                ctr += 1
            # grab residue and atom for key accesing
            aaa = row.residue_name if row.residue_name in to1letter.keys() else "GLY"
            aa_atom = row.atom_name
            if aaa == "GLY" and aa_atom not in ["CA", "N", "O", "C"]:
                continue
            key = (aaa, aa_atom)
            # Extract positional information
            xyz[ctr, aa2idx[key], 0] = row.x_coord
            xyz[ctr, aa2idx[key], 1] = row.y_coord
            xyz[ctr, aa2idx[key], 2] = row.z_coord
            # Extract occupancy
            occupancy[ctr, aa2idx[key]] = row.occupancy
            # Attend to info
            mask[ctr, aa2idx[key]] = 1.0
        # Reformat the 0.0s to nans like how they have originally
        xyz = torch.where(xyz == 0., float('nan'), xyz)
        # Create bfac 
        bfac = torch.where(mask==0., float('nan'), mask)
        bfac = torch.where(bfac==1., float(0.), bfac)

        # Write data
        OUT = os.path.join(OUTPUT,filename)

        # create dict to write
        keys = ['seq', 'xyz', 'mask', 'bfac', 'occ']
        vals = [seq, xyz, mask, bfac, occupancy]
        out_dict = dict(zip(keys, vals))
        
        torch.save(out_dict, f"{OUT}_{id}.pt")

        # Add seq to seq_all
        seq_all.append(seq)

    return seq_all

def write_pt_general(f: object, OUTPUT:str, seq:list) -> int:
    """Write out the general generic monomeric .pt file for training

    PARAMETERS
    ----------
    f: file object
        PDB file

    OUTPUT: str
        OUTPUT dir path

    seq: list
        list of one letter string of AAs in the peptide.
        This works as an input, because the 

    RETURNS
    -------
    0: int
        Zero status exit
    """
    # Strip filename path info
    filename = f.strip(".pdb").split("/")[-1]

    # Combine output and filename to create out name
    OUT = os.path.join(OUTPUT, filename+".pt")

    # correct seq input
    if len(seq) == 1:
        seq = [[seq[0], seq[0]]]

    # Specify hard code contents
    contents = {
        'method': 'IN-SILICO',
        'date': '2023-08-16',
        'resolution': None,
        'chains': ['A'],
        'seq': seq,
        'id': filename,
        'asmb_chains': ['A'],
        'asmb_details': ['author_defined_assembly'],
        'asmb_method': ['?'],
        'asmb_ids': ['1'],
        'asmb_xform0': torch.tensor([[[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [0., 0., 0., 1.]]]),
        'tm': torch.tensor([[[1., 1., 0.]]])
    }

    # write to .pt
    torch.save(contents, OUT)

    return 0
    

def main():
    """Main function that cycles through a pdb input dir and returns .pt
    files for each of them in the correct orientation of the .pt files 
    needed to train the model."""

    # Define argument parser
    p = argparse.ArgumentParser("Arguments Needed to Create .pt Files")
    p.add_argument("--path_to_pdb_dir", type=str, help="Specify Path to directory with noncanonical PDBs \
    (default = ./noncanonical_pdbs)", default="./noncanonical_pdbs")
    p.add_argument("--path_to_output", type=str, help="Specify Path to output directory \
    (default = ../training_data_repo)", default="../training_data_repo")
    PARSER = p.parse_args()

    # Specify passed argument global variables
    INPUT = PARSER.path_to_pdb_dir
    OUTPUT = PARSER.path_to_output
    RES_NAMES = [
        'ALA','ARG','ASN','ASP','CYS', # L
        'GLN','GLU','GLY','HIS','ILE', # L
        'LEU','LYS','MET','PHE','PRO', # L
        'SER','THR','TRP','TYR','VAL', # L
        'DAL','DAR','DAN','DAS','DCY', # D
        'DGL','DGU','DGN','DHI','DIL', # D
        'DLE','DLY','DME','DPH','DPR', # D
        'DSE','DTH','DTR','DTY','DVA'  # D
    ]

    # Init L single AA code
    RES_NAMES_1 = 'ARNDCQEGHILKMFPSTWYV' # L 
    RES_NAMES_2 = RES_NAMES_1.lower() # D version
    RES_NAMES_TOTAL = RES_NAMES_1 + RES_NAMES_2 # Combine into one string

    # generate Dictionaries for 1 to three and three to 1 conversion
    to1letter = {aaa:a for a,aaa in zip(RES_NAMES_TOTAL,RES_NAMES)}
    to3letter = {a:aaa for a,aaa in zip(RES_NAMES_TOTAL,RES_NAMES)}

    ATOM_NAMES = [
        ("N", "CA", "C", "O", "CB"), # ala
        ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"), # arg
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"), # asn
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"), # asp
        ("N", "CA", "C", "O", "CB", "SG"), # cys
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"), # gln
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"), # glu
        ("N", "CA", "C", "O"), # gly
        ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"), # his
        ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"), # ile
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"), # leu
        ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"), # lys
        ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"), # met
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"), # phe
        ("N", "CA", "C", "O", "CB", "CG", "CD"), # pro
        ("N", "CA", "C", "O", "CB", "OG"), # ser
        ("N", "CA", "C", "O", "CB", "OG1", "CG2"), # thr
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"), # trp
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"), # tyr
        ("N", "CA", "C", "O", "CB", "CG1", "CG2"), # val
        # Repeat of above for D 
        ("N", "CA", "C", "O", "CB"), # ala
        ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"), # arg
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"), # asn
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"), # asp
        ("N", "CA", "C", "O", "CB", "SG"), # cys
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"), # gln
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"), # glu
        ("N", "CA", "C", "O"), # gly
        ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"), # his
        ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"), # ile
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"), # leu
        ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"), # lys
        ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"), # met
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"), # phe
        ("N", "CA", "C", "O", "CB", "CG", "CD"), # pro
        ("N", "CA", "C", "O", "CB", "OG"), # ser
        ("N", "CA", "C", "O", "CB", "OG1", "CG2"), # thr
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"), # trp
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"), # tyr
        ("N", "CA", "C", "O", "CB", "CG1", "CG2") # val
    ]
            
    idx2ra = {(RES_NAMES_TOTAL[i],j):(RES_NAMES[i],a) for i in range(40) for j,a in enumerate(ATOM_NAMES[i])} # Change to 40, since D added

    aa2idx = {(r,a):i for r,atoms in zip(RES_NAMES,ATOM_NAMES) 
              for i,a in enumerate(atoms)}
    aa2idx.update({(r,'OXT'):3 for r in RES_NAMES})

    # Loop through files within the inputs
    file_pattern = os.path.join(INPUT,'*.pdb')
    files = glob.glob(file_pattern)
    for file in tqdm(files):
        seq_list = extract_pdb_info(
            f=file,
            to1letter=to1letter,
            aa2idx=aa2idx,
            OUTPUT=OUTPUT,
            res_names=RES_NAMES,
        )
        write_pt_general(
            f=file,
            OUTPUT=OUTPUT,
            seq=seq_list,
        )

    return 0


###############################################################
########################## Main Logic #########################
###############################################################


if __name__ == "__main__":
    main()

