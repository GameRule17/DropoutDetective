import pandas as pd
import os
import sys
from pyswip import Prolog

# Creazione di una nuova istanza Prolog
prolog = Prolog()

def facts_from_dataframe(data):
    # Definizione dei fatti per ogni studente nel dataset
    for index, row in data.iterrows():
        fact = (f"student({index}, '{row['Marital status']}', '{row['Application mode']}', {row['Application order']}, "
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
        
        prolog.assertz(fact)

        # Stampa solo l'ultima asserzione
        # if index == data.index[-1]:
            # print(f"Asserting fact: {fact}")

def new_features_extraction(data):
    # Definizione di nuove feature usando regole Prolog

    # Financially Stable: studente che non è un debitore ed è in regola con il pagamento delle tasse
    prolog.assertz("financially_stable(ID) :- student(ID, _, _, _, _, _, _, _, _, _, _, _, _, _, Debtor, FeesUpToDate, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _), (Debtor =:= 0, FeesUpToDate =:= 1)")

    # Interaction features per performance accademiche                                                                                              #24                           #30
    prolog.assertz("interaction_cu_1st_2nd_approved(ID, Result) :- student(ID, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, CU1stApproved, _, _, _, _, _, CU2ndApproved, _, _, _, _, _), Result is CU1stApproved * CU2ndApproved")
                                                                                                                                                 #25                        #31
    prolog.assertz("interaction_cu_1st_2nd_grade(ID, Result) :- student(ID, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, CU1stGrade, _, _, _, _, _, CU2ndGrade, _, _, _, _), Result is CU1stGrade * CU2ndGrade")

    # Feature Aggregate per performance accademiche
    prolog.assertz("total_cu_approved(ID, Result) :- student(ID, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, CU1stApproved, _, _, _, _, _, CU2ndApproved, _, _, _, _, _), Result is CU1stApproved + CU2ndApproved")
    prolog.assertz("total_cu_grade(ID, Result) :- student(ID, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, CU1stGrade, _, _, _, _, _, CU2ndGrade, _, _, _, _), Result is (CU1stGrade + CU2ndGrade) / 2")

    # Esecuzione query per ottenere i nuovi attributi per ogni studente
    financially_stable_students = list(prolog.query("financially_stable(ID)"))
    interaction_cu_1st_2nd_approved = list(prolog.query("interaction_cu_1st_2nd_approved(ID, Result)"))
    interaction_cu_1st_2nd_grade = list(prolog.query("interaction_cu_1st_2nd_grade(ID, Result)"))
    total_cu_approved = list(prolog.query("total_cu_approved(ID, Result)"))
    total_cu_grade = list(prolog.query("total_cu_grade(ID, Result)"))

    # Conversione dei risultati delle query
    financially_stable_students_ids = [student['ID'] for student in financially_stable_students]
    interaction_cu_1st_2nd_approved_dict = {res['ID']: res['Result'] for res in interaction_cu_1st_2nd_approved}
    interaction_cu_1st_2nd_grade_dict = {res['ID']: res['Result'] for res in interaction_cu_1st_2nd_grade}
    total_cu_approved_dict = {res['ID']: res['Result'] for res in total_cu_approved}
    total_cu_grade_dict = {res['ID']: res['Result'] for res in total_cu_grade}

    # Inserimento delle nuove feature nel dataframe
    data['Financially Stable'] = data.index.isin(financially_stable_students_ids).astype(int)
    data['Interaction_CU_1st_2nd_Approved'] = data.index.map(interaction_cu_1st_2nd_approved_dict)
    data['Interaction_CU_1st_2nd_Grade'] = data.index.map(interaction_cu_1st_2nd_grade_dict)
    data['Total_CU_Approved'] = data.index.map(total_cu_approved_dict)
    data['Total_CU_Grade'] = data.index.map(total_cu_grade_dict)

def kb_feature_engineering(data):
    facts_from_dataframe(data)
    new_features_extraction(data)