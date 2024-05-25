
def preprocessing(data):
    # Visualizzazione delle prime righe del dataset
    # print(data.head())

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

    # Visualizza solo i nomi delle colonne
    # print(data.columns)