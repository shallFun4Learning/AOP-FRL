from Bio import SeqIO
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import tensor
from sklearn.model_selection import train_test_split
import sklearn
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModel

import pandas as pd
from rdkit import Chem
import random

input_csv = '/data/xuxf/antiO_p70/data/03_p90_AO_db.csv'
'''
Enter the CSV file address.
Please note that we recommend changing the FRS 
column to label to conform to the subsequent interface.
'''

def generate_output_path(input_csv):
    input_dir, input_filename = os.path.split(input_csv)
    input_name, input_ext = os.path.splitext(input_filename)
    output_name = input_name + '_smiles_p' + input_ext
    output_csv = os.path.join(input_dir, output_name)
    return output_csv



standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

standard_amino_acids_list = list(standard_amino_acids)

def sequence_to_smiles(sequence):
    non_standard_count = 0
    new_sequence = ""
    
    for residue in sequence:
        if residue not in standard_amino_acids:
            residue = random.choice(standard_amino_acids_list)
            non_standard_count += 1
        new_sequence += residue

    mol = Chem.MolFromSequence(new_sequence)
    
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        return smiles, non_standard_count
    else:
        return None, non_standard_count

df = pd.read_csv(input_csv)

total_sequences = len(df)
successful_conversions = 0
total_non_standard_residues = 0

smiles_list = []
for sequence in df['sequence']:
    smiles, non_standard_count = sequence_to_smiles(sequence)
    smiles_list.append(smiles)
    total_non_standard_residues += non_standard_count
    if smiles is not None:
        successful_conversions += 1

df['SMILES'] = smiles_list

output_csv = generate_output_path(input_csv)
df.to_csv(output_csv, index=False)

# Print statistical results
print(f"Total number of input sequences: {total_sequences}")
print(f"Number of sequences successfully converted: {successful_conversions}")
print(f"Number of non-standard amino acids found and processed: {total_non_standard_residues}")
print(f"Smiles sequence saved to: {output_csv}")

