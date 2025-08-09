import pandas as pd
from rdkit import Chem

# Load the CSV file
data = pd.read_csv('TC.csv')

# Function to count the number of atoms from SMILES
def count_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return mol.GetNumAtoms()
    else:
        return 0  # Handle invalid SMILES gracefully

# Apply the atom counting function to the SMILES column
data['atom_count'] = data['smiles'].apply(count_atoms)

# Count how many molecules have over 50 atoms
num_over_50_atoms = (data['atom_count'] > 50).sum()

print(f'Number of molecules with more than 50 atoms: {num_over_50_atoms}')

# Remove molecules with more than 50 atoms
data = data[data['atom_count'] <= 50]

# Output the filtered data
data.to_csv('filtered_TC_SC_with_scscore.csv', index=False)

print(f'Number of remaining molecules: {len(data)}')
