import pandas as pd

# Import delle funzioni da me definite
from preprocessing import *
from knowledge_base import *
from supervised import *
from bayesian_network import *

# Import del dataset
data = pd.read_csv("dataset/dataset.csv")

# ATTENZIONE: Si consiglia di eseguire ogni blocco singolarmente!

initial_preprocessing(data)

print(data.head())

# Supervised learning con una prima versione di dataset preprocessato
train_model_k_fold(data, "Target", "Dataset Originale")

######################################################################

kb_feature_engineering(data)

print(data.corr()['Target'].sort_values(ascending=False))

after_kb_feature_engineering_preprocessing(data)

print(data.head())

# Supervised learning su nuova versione di dataset processato
train_model_k_fold(data, "Target", "Feature Engineering")

######################################################################

# Supervised learning con SMOTE
train_model_k_fold(data, "Target", "SMOTE", True)

######################################################################

merge_target_feature(data)

# Supervised learning con SMOTE e nuova feature Target Binaria
train_model_k_fold(data, "Target", "SMOTE + Binary", True)

######################################################################

# Supervised learning con SMOTE + Binary ma con VotingClassifier
ensemble_model(data, "Target", "SMOTE + Binary", True)

######################################################################

# Creazione della Rete Bayesiana su dataset semplificato
simplify_dataset_for_bayesian_network(data)
discretize_dataset(data)

# Crea un nuovo file csv con il nuovo dataset
#data.to_csv("dataset/dataset_simplified_bn.csv", index=False)
bn = create_load_bayesian_network(data)

# Genera un esempio randomico e predici il valore di Target
example = generateRandomExample(bn)
print("Esempio randomico:")
print(example)

# Conversione del dataframe in un dizionario per la predizione
predict(bn, example.to_dict('records')[0], 'Target')

# Rimozione di una feature dall'esempio e predizione del valore di Target
del(example['Gender'])
print("Esempio randomico senza Gender:")
print(example)

predict(bn, example.to_dict('records')[0], 'Target')

# Valutazione della rete bayesiana
print(correlation_score(bn, data, score=balanced_accuracy_score))

# Query sulla rete bayesiana
infer = VariableElimination(bn)
query_report(infer, variables=['Scholarship holder'], evidence={'Financially Stable': 1},
             desc='Data la osservazione che uno studente è stabile dal punto di vista finanziazio'
                  ' qual è la distribuzione di probabilità per Scholarship holder')

query_report(infer, variables=['Financially Stable'], evidence={'Age': 4.0},
             desc='Data la osservazione che uno studente ha una età avanzata (valore più alto discretizzato)'
                  ' qual è la distribuzione di probabilità per Financially Stable')
