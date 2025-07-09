import pandas as pd
from rdkit import Chem
from tqdm import tqdm

print("Loading dataset...")
df = pd.read_csv('data/raw/TC.csv.gz')

max_atoms = 0
largest_smiles = ""

print("Analyzing molecules...")
for smiles in tqdm(df['smiles']):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            if num_heavy_atoms > max_atoms:
                max_atoms = num_heavy_atoms
                largest_smiles = smiles
    except:
        print(f"Could not process SMILES: {smiles}")

print("\n--- Analysis Complete ---")
print(f"The largest molecule has {max_atoms} heavy atoms.")
print(f"SMILES: {largest_smiles}")
print("-----------------------")
