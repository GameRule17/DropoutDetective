import pandas as pd
import os
import sys

# Aggiungi il percorso di SWI-Prolog al PATH
swi_prolog_path = r"C:\Program Files\swipl\bin"
os.environ['PATH'] += os.pathsep + swi_prolog_path

from pyswip import Prolog

# Creazione di una nuova istanza Prolog
prolog = Prolog()

def facts_from_dataframe(data):
    # Definizione dei fatti per ogni studente nel dataset
    for index, row in data.iterrows():
        prolog.assertz(f"student({index}, '{row['Marital status']}', '{row['Application mode']}', {row['Application order']}, "
                    f"'{row['Course']}', '{row['Daytime/evening attendance']}', '{row['Previous qualification']}', "
                    f"'{row['Nationality']}', '{row['Mother qualification']}', '{row['Father qualification']}', "
                    f"'{row['Mother occupation']}', '{row['Father occupation']}', {row['Displaced']}, "
                    f"{row['Educational special needs']}, {row['Debtor']}, {row['Tuition fees up to date']}, "
                    f"'{row['Gender']}', {row['Scholarship holder']}, {row['Age']}, {row['International']}, "
                    f"{row['Curricular units 1st sem (credited)']}, {row['Curricular units 1st sem (enrolled)']}, "
                    f"{row['Curricular units 1st sem (evaluations)']}, {row['Curricular units 1st sem (approved)']}, "
                    f"{row['Curricular units 1st sem (grade)']}, {row['Curricular units 1st sem (without evaluations)']}, "
                    f"{row['Curricular units 2nd sem (credited)']}, {row['Curricular units 2nd sem (enrolled)']}, "
                    f"{row['Curricular units 2nd sem (evaluations)']}, {row['Curricular units 2nd sem (approved)']}, "
                    f"{row['Curricular units 2nd sem (grade)']}, {row['Curricular units 2nd sem (without evaluations)']}, "
                    f"{row['Unemployment rate']}, {row['Inflation rate']}, {row['GDP']})")


def new_features_extraction(data):
    # Definizione di nuove feature usando regole Prolog

    # Buon Studente: Approva almeno l'80% delle unità curriculari iscritte
    prolog.assertz("""
    good_student(ID) :-
        student(ID, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, Enrolled, _, Approved),
        Enrolled > 0,
        Approved >= 0.8 * Enrolled.
    """)

    # Finanziariamente Stabile: Non è un debitore e le tasse universitarie sono aggiornate
    prolog.assertz("""
    financially_stable(ID) :-
        student(ID, _, _, _, _, _, _, _, _, _, _, _, _, _, Debtor, FeesUpToDate, _, _, _, _, _, _, _, _),
        Debtor == 0,
        FeesUpToDate == 1.
    """)

    # Esecuzione query per ottenere i nuovi attributi per ogni studente
    good_students = list(prolog.query("good_student(ID)"))
    financially_stable_students = list(prolog.query("financially_stable(ID)"))

    # Conversione dei risultati delle query in un formato più utile
    good_students_ids = [student['ID'] for student in good_students]
    financially_stable_students_ids = [student['ID'] for student in financially_stable_students]

    # Inserimento delle nuove feature nel dataframe
    data['Good Student'] = data.index.isin(good_students_ids).astype(int)
    data['Financially Stable'] = data.index.isin(financially_stable_students_ids).astype(int)

