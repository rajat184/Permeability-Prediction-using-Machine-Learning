from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import pandas as pd

import numpy as np

url = 'C:/Users/Suraj/Desktop/IP sem 6/test/Codes/Lazy Predict/input_data.csv'
dataset = pd.read_csv(url)
dataset.shape

def canonical_smiles(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles] 
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return smiles

from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

Chem.MolFromSmiles('C=CCC')
Chem.MolFromSmiles('CCC=C')

canonical_smiles(['C=CCC'])
canonical_smiles(['CCC=C'])

# Canonical SMILES
Canon_SMILES = canonical_smiles(dataset.SMILES)
len(Canon_SMILES)

# Put the smiles in the dataframe
dataset['SMILES'] = Canon_SMILES
dataset

# Create a list for duplicate smiles
duplicates_smiles = dataset[dataset['SMILES'].duplicated()]['SMILES'].values
len(duplicates_smiles)

# Create a list for duplicate smiles
dataset[dataset['SMILES'].isin(duplicates_smiles)].sort_values(by=['SMILES'])

dataset_new = dataset.drop_duplicates(subset=['SMILES'])

def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles] 
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:
        # add hydrogens to molecules
        mol=Chem.AddHs(mol)
        # Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names 

def generate_costas_waveform(N):
    # Generate random binary sequence
    data = np.random.randint(0, 2, N)
    
    # Generate Costas sequence
    costas_sequence = np.zeros(N)
    for i in range(N):
        costas_sequence[i] = (data[i] + 2 * costas_sequence[i - 1]) % 2
        
    return costas_sequence

# Function call
Mol_descriptors,desc_names = RDkit_descriptors(dataset_new['SMILES'])

# df_with_200_descriptors
df_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
df_with_200_descriptors.to_csv('descriptors2.csv', index=False)

