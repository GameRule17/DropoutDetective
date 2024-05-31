from sklearn.preprocessing import KBinsDiscretizer

def initial_preprocessing(data):
    # Rinominazione di alcune colonne con nomi errati o troppo esplicativi
    data.rename(columns = {'Nacionality':'Nationality', 'Age at enrollment':'Age'}, inplace = True)

    # Rimozione dai nomi delle colonne il genitivo sassone
    data.columns = data.columns.str.replace("'s", "")

    # Verifica della presenza di valori nulli
    if data.isnull().any().any():
        raise ValueError("Il dataset contiene valori nulli.")

    # Verifica della presenza di valori duplicati
    if data.duplicated().any():
        raise ValueError("Il dataset contiene valori duplicati.")
    
    # Codifica della feature target poiché è l'unico campo non numerico nel dataset
    # print(data["Target"].unique())
    # Ci sono 3 valori differenti nella colonna Target, quindi possiamo codificarli con un dizionario
        # Dropout - 0
        # Enrolled - 1
        # Graduate - 2
    data['Target'] = data['Target'].map({
        'Dropout':0,
        'Enrolled':1,
        'Graduate':2
    })
    # print(data["Target"].unique())

# Funzione che mostra la correlazione tra le feature e la feature target
def show_feature_correlation(data):
    print(data.corr()['Target'].sort_values(ascending=False))

# Funzione che rimuove le colonne con bassa correlazione rispetto alla feature target
def drop_colums_with_low_correlation(data):
    data.drop(columns=['Nationality', 
                        'Mother qualification', 
                        'Father qualification', 
                        'Educational special needs', 
                        'International', 
                        'Curricular units 1st sem (without evaluations)',
                        'Curricular units 1st sem (credited)',
                        'Unemployment rate', 
                        'Inflation rate',
                        'Course',
                        'Mother occupation', 
                        'Father occupation',
                        'GDP'], axis=1, inplace=True)

# Funzione che rimuove le colonne dopo il feature engineering con Knowledge Base
def drop_colums_after_kb_feature_engineering(data):
    data.drop(columns=['Curricular units 1st sem (approved)', 
                        'Curricular units 2nd sem (approved)',
                        'Curricular units 1st sem (grade)', 
                        'Curricular units 2nd sem (grade)',
                        'Debtor',
                        'Tuition fees up to date',
                        ], axis=1, inplace=True)
    
def after_kb_feature_engineering_preprocessing(data):
    drop_colums_with_low_correlation(data)
    drop_colums_after_kb_feature_engineering(data)

# Funzione che effettua il merge della feature target in una nuova feature binaria
def merge_target_feature(data):
    # Nuova feature target binaria, unendo gli stati Enrolled e Graduate nella categoria No Dropout.
    data['Target'] = data['Target'].map(lambda x: 1 if x == 0 else 0)

# Funzione che discretizza il dataset
def discretize_dataset(data):
    discretizer = KBinsDiscretizer(encode='ordinal', strategy='uniform')
    continuos_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[continuos_columns] = discretizer.fit_transform(data[continuos_columns])

# Funzione che semplifica il dataset per la creazione della rete bayesiana
def simplify_dataset_for_bayesian_network(data):
    data.drop(columns=['Displaced',
                        'Curricular units 2nd sem (evaluations)', 
                        'Application order', 
                        'Daytime/evening attendance', 
                        'Curricular units 2nd sem (credited)',
                        'Marital status',
                        'Previous qualification',
                        'Curricular units 2nd sem (without evaluations)',
                        ], axis=1, inplace=True)
