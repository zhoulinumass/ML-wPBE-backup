# %%

import csv
import os
import rdkit
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from joblib import dump, load
import numpy as np
import lightgbm as lgb
from sklearn import preprocessing
import argparse
# read smiles of molecules
import time

parser = argparse.ArgumentParser(description='Input smi file path')
parser.add_argument('--smi', default='Put_Prediction_File_Here/outliers.smi', type=str)
args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")
file = open(args.smi)

smiles = []
for line in file:
    smiles.append(line.strip('\n'))
file.close()
t = time.time()
# data for RDKit Des
mols = []
for i in range(0, len(smiles)):
    q = i + 1
    mols.append(Chem.MolFromSmiles(smiles[i]))

smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
descs = [desc_name[0] for desc_name in Descriptors._descList]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
descriptors = pd.DataFrame([desc_calc.CalcDescriptors(mol) for mol in mols])
descriptors.columns = descs
descriptors.index = smiles_list
index_list = list(map(str, list(range(len(mols)))))
y = pd.DataFrame(index_list)
y.index = smiles_list
y.columns = ["index"]
dataset = pd.concat([y, descriptors], axis=1)
dataset.to_csv('Put_Prediction_File_Here/Rdkit_Descriptor.csv')

# data for RdKit FP
with open('Put_Prediction_File_Here/FP_Morgan2.csv', 'w', newline='') as f:
    f_csv = csv.writer(f)
    for sm_num in range(0, len(smiles)):
        bits = []
        mol = Chem.MolFromSmiles(smiles[sm_num])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        bit = fp.ToBitString()
        for i in range(0, 2048):
            bits.append(bit[i])
        f.write(smiles[sm_num])
        f.write(',')
        f_csv.writerow(bits)
        bits.clear()


# Read Several Files, load as x1(des1), x2(des2), name.
def read_file_descriptor(file_path, n=1, startline=1):
    x_input = []
    jishu = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            jishu += 1
            if jishu > startline:
                x_input.append(row[n:])
    x_input = np.array(x_input)
    for i in range(0,len(x_input)):
        try:
            x_input[i] = x_input[i].astype('float')
        except:
            for q in range(0,len(x_input[i])):
                try:
                    x_input[i][q] = x_input[i][q].astype('float')
                except:
                    x_input[i][q] = float(0)
    x_input = x_input.astype('float')
    return x_input


def read_file_name(file_path):
    name = []
    file = open(file_path)
    for line in file:
        name.append(line.strip('\n'))
    file.close()
    return name

# Load trained model.
def load_model_predict(model_file, x_dataset):
    reg_layer1 = load(model_file)
    x_pre = reg_layer1.predict(x_dataset)
    return x_pre

x_des = read_file_descriptor(file_path='Put_Prediction_File_Here/Rdkit_Descriptor.csv', n=2, startline=1)
x_Morgan2 = read_file_descriptor(file_path='Put_Prediction_File_Here/FP_Morgan2.csv', n=1, startline=0)

name = smiles
x2 = np.concatenate((x_des, x_Morgan2), axis=1)[:, 0:]





x_pre_first_layer = []
print('Predicton Begin')
x_pre_first_layer.append(load_model_predict('wholedataset_Model/lgb_des2.pkl', x2))
x_pre_first_layer.append(load_model_predict('wholedataset_Model/xgboost_des2.pkl', x2))
x_pre_first_layer.append(load_model_predict('wholedataset_Model/GBRT_des2.pkl', x2))
x_pre_first_layer.append(load_model_predict('wholedataset_Model/Lasso_des2.pkl', x2))
x_pre_first_layer.append(load_model_predict('wholedataset_Model/RF_des2.pkl', x2))
print('Prediction Finish')

x_pre_first_layer_T = list(zip(*x_pre_first_layer))
x_pre_first_layer_T = np.array(x_pre_first_layer_T)

reg_second_layer = load('wholedataset_Model/Lars_secondlayer.pkl')
x_pre_second_layer = reg_second_layer.predict(x_pre_first_layer_T)
assert len(smiles) == len(x_pre_second_layer)
for i in range(0, len(name)):
    print(round( x_pre_second_layer[i]),smiles[i])
print("Time consumed: %.3fs" % (time.time() - t))
# %%


# %%
