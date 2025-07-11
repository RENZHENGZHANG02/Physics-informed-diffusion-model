import math, os
import pickle
import os.path as op
from generate_registry import load_registry
import pathlib

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score


from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
rdBase.DisableLog('rdApp.error')

class TaskModel():
    """Scores based on an ECFP classifier."""
    def __init__(self, model_path, task_name):
        base_path = pathlib.Path(os.path.realpath(__file__))
        self.registry = load_registry(os.path.join(base_path.parents[2], "configs/task_registry.yaml"))
        
        task_type = self.registry[task_name]["types"]
        self.task_name = task_name
        self.task_type = task_type
        self.model_path = model_path
        self.metric_func = roc_auc_score if 'classification' in self.task_type else mean_absolute_error
        clf_model_path = model_path.replace('.pkl', '_clf.pkl')
        reg_model_path = model_path.replace('.pkl', '_reg.pkl')
        self.classifier = None
        self.regressor = None

        try:
            self.model = load(model_path)
            print(self.task_name, ' evaluator loaded')
        except:
            print(self.task_name, ' evaluator not found, training new one...')
            if 'classification' in task_type:
                self.classifier = RandomForestClassifier(random_state=0)
                dump(self.classifier, clf_model_path)
            if 'regression' in task_type:
                self.regressor = RandomForestRegressor(random_state=0)
                dump(self.regressor, reg_model_path)
            perfermance = self.train()
            print('Oracle peformance: ', perfermance)

    def train(self):
        data_path = os.path.dirname(self.model_path)
        data_path = os.path.join(os.path.dirname(self.model_path), '..', f'raw/{self.task_name}.csv.gz')
        df = pd.read_csv(data_path)
        
        col_name_regression = []
        col_name_classification = []
        for t, col in zip(self.registry[self.task_name]['types'], self.registry[self.task_name]['cols']):
            if t == 'regression':
                col_name_regression.append(col)
            elif t == 'classification':
                col_name_classification.append(col)

        df = df[df[col_name_regression+col_name_classification].notna().all(axis=1)]
        y_regress = df[col_name_regression].to_numpy()
        y_classify = df[col_name_classification].to_numpy()
        x_smiles = df['smiles'].to_numpy()
        #mask = ~np.isnan(y)
        #y = y[mask]

        y_classify = y_classify.astype(int)

        #x_smiles = x_smiles[mask]
        x_fps = []
        mask = []
        for i,smiles in enumerate(x_smiles):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = TaskModel.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            x_fps.append(fp)
        x_fps = np.concatenate(x_fps, axis=0)

        regress_perf = 0.0
        classify_perf = 0.0
        if self.regressor is not None:
            self.regressor.fit(x_fps, y_regress)
            y_pred = self.regressor.predict(x_fps)
            regress_perf = self.metric_func(y_regress, y_pred)
            print(f'{self.task_name} regression performance: {regress_perf}')
            dump(self.classifier, self.model_path.replace('.pkl', '_clf.pkl'))
        else:
            print("There is no regressor for the task")
        
        if self.classifier is not None:
            self.classifier.fit(x_fps, y_classify)
            y_pred = self.classifier.predict(x_fps)
            classify_perf = self.metric_func(y_classify, y_pred)
            print(f'{self.task_name} classification performance: {classify_perf}')
            dump(self.regressor, self.model_path.replace('.pkl', '_reg.pkl'))
        else:
            print("There is no classifier for the task")
        return classify_perf, regress_perf

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = TaskModel.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        mask = np.array(mask)

        outputs = {}

        if self.classifier is not None:
            scores_cls = self.classifier.predict_proba(fps)[:, 1]
            outputs['classification'] = (scores_cls * mask).astype(np.float32)

        if self.regressor is not None:
            scores_reg = self.regressor.predict(fps)
            outputs['regression'] = (scores_reg * mask).astype(np.float32)

        return outputs

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

###### SAS Score ######
_fscores = None

def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateSAS(smiles_list):
    scores = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        score = calculateScore(mol)
        scores.append(score)
    return np.float32(scores)

def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore
