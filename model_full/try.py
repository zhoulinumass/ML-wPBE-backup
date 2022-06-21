import csv
import os
import rdkit
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from padelpy import padeldescriptor

inputfile_smiles='Put_Prediction_File_Here/example.smi'

#read smiles of molecules 
file = open(inputfile_smiles)
smiles = []
for line in file:
    if len(line.strip('\n'))>2:
        smiles.append(line.strip('\n'))
file.close()

#data for PaDEL
try:
    padeldescriptor(config='./descriptors.xml',maxruntime=100000,retainorder=True,standardizenitro=True,detectaromaticity=True)
    padeldescriptor(mol_dir=inputfile_smiles, d_file='./Put_Prediction_File_Here/PaDEL_FP.csv', fingerprints=True)
except:
    print('padel have problem')

