import pandas as pd

# Import delle funzioni da me definite
from preprocessing import *
# from knowledge_base import *
from supervised import *

# Import del dataset
data = pd.read_csv("dataset/dataset.csv")

preprocessing(data)
#facts_from_dataframe(data)
#new_features_extraction(data)

# Visualizzazione delle prime righe del dataset 
# print(data.head())

# Supervised learning con una prima versione di dataset preprocessato
supervised(data, "Target")