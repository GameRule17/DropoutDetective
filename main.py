import pandas as pd

# Import delle funzioni da me definite
from preprocessing import *
from knowledge_base import *
from supervised import *

# Import del dataset
data = pd.read_csv("dataset/dataset.csv")

initial_preprocessing(data)

# print(data.head())

# Supervised learning con una prima versione di dataset preprocessato
# train_model_k_fold(data, "Target", "Dataset Originale")

kb_feature_engineering(data)

# print(data.corr()['Target'].sort_values(ascending=False))

after_kb_feature_engineering_preprocessing(data)

# Visualizzazione delle prime righe del dataset 
# print(data.head())

# Supervised learning su nuova versione di dataset processato
# train_model_k_fold(data, "Target", "Feature Engineering")

# Supervised learning con SMOTE
train_model_k_fold(data, "Target", "SMOTE", True)