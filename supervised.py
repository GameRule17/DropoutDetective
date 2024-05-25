# Import librerie utilizzate
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from xgboost import XGBClassifier
from sklearn import svm

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# Definizione della funzione sturgeRule
def sturgeRule(n):
    return int(1 + 3.322 * np.log10(n))

def return_best_hyperparameters(dataset, target):
    # Cross Validation Strategy (Repeated Stratified K-Fold) with 5 splits and 2 repeats and a random state of 42 for reproducibility
    X = dataset.drop(target, axis=1)
    y = dataset[target]

    # Splitting the dataset into the Training set and Test set (80% training, 20% testing) with stratification and shuffling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=42
    )

    CV = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)

    # Models Evaluated
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    lr = LogisticRegression()

    # Hyperparameters for each model
    DecisionTreeHyperparameters = {
        "DecisionTree__criterion": ["gini", "entropy", "log_loss"],
        "DecisionTree__max_depth": [5, 10, 20, 40],
        "DecisionTree__min_samples_split": [2, 5, 10, 20],
        "DecisionTree__min_samples_leaf": [2, 5, 10, 20],
        "DecisionTree__splitter": ["best"],
    }
    RandomForestHyperparameters = {
        "RandomForest__criterion": ["gini", "entropy", "log_loss"],
        "RandomForest__n_estimators": [10, 100, 200],
        "RandomForest__max_depth": [5, 10, 20],
        "RandomForest__min_samples_split": [2, 5, 10],
        "RandomForest__min_samples_leaf": [2, 5, 10],
    }
    LogisticRegressionHyperparameters = {
        "LogisticRegression__penalty": ["l1", "l2", "elasticnet", "none"],
        "LogisticRegression__C": [0.1, 1, 10],
        "LogisticRegression__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    }

    # GridSearchCV for each model with the respective hyperparameters
    gridSearchCV_dtc = GridSearchCV(
        Pipeline([("DecisionTree", dtc)]),
        DecisionTreeHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    gridSearchCV_rfc = GridSearchCV(
        Pipeline([("RandomForest", rfc)]),
        RandomForestHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    gridSearchCV_lr = GridSearchCV(
        Pipeline([("LogisticRegression", lr)]),
        LogisticRegressionHyperparameters,
        cv=CV,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    # Fitting the models with the training data
    gridSearchCV_dtc.fit(X_train, y_train)
    gridSearchCV_rfc.fit(X_train, y_train)
    gridSearchCV_lr.fit(X_train, y_train)

    # Returning the best hyperparameters for each model
    bestParameters = {
        "DecisionTree__criterion": gridSearchCV_dtc.best_params_[
            "DecisionTree__criterion"
        ],
        "DecisionTree__max_depth": gridSearchCV_dtc.best_params_[
            "DecisionTree__max_depth"
        ],
        "DecisionTree__min_samples_split": gridSearchCV_dtc.best_params_[
            "DecisionTree__min_samples_split"
        ],
        "DecisionTree__min_samples_leaf": gridSearchCV_dtc.best_params_[
            "DecisionTree__min_samples_leaf"
        ],
        "RandomForest__n_estimators": gridSearchCV_rfc.best_params_[
            "RandomForest__n_estimators"
        ],
        "RandomForest__max_depth": gridSearchCV_rfc.best_params_[
            "RandomForest__max_depth"
        ],
        "RandomForest__min_samples_split": gridSearchCV_rfc.best_params_[
            "RandomForest__min_samples_split"
        ],
        "RandomForest__min_samples_leaf": gridSearchCV_rfc.best_params_[
            "RandomForest__min_samples_leaf"
        ],
        "RandomForest__criterion": gridSearchCV_rfc.best_params_[
            "RandomForest__criterion"
        ],
        "LogisticRegression__penalty": gridSearchCV_lr.best_params_[
            "LogisticRegression__penalty"
        ],
        "LogisticRegression__C": gridSearchCV_lr.best_params_[
            "LogisticRegression__C"
        ],
        "LogisticRegression__solver": gridSearchCV_lr.best_params_[
            "LogisticRegression__solver"
        ],
    }
    return bestParameters, X_train, y_train, X_test, y_test

def save_best_hyperparameters(bestParameters):
    # Prepara il testo da salvare
    text = "\n".join([
        "Migliori parametri per Decision Tree Classifier:",
        f"  Criterion: {bestParameters['DecisionTree__criterion']}",
        f"  Max Depth: {bestParameters['DecisionTree__max_depth']}",
        f"  Min Samples Split: {bestParameters['DecisionTree__min_samples_split']}",
        f"  Min Samples Leaf: {bestParameters['DecisionTree__min_samples_leaf']}",
        "\nMigliori parametri per Random Forest Classifier:",
        f"  Criterion: {bestParameters['RandomForest__criterion']}",
        f"  N Estimators: {bestParameters['RandomForest__n_estimators']}",
        f"  Max Depth: {bestParameters['RandomForest__max_depth']}",
        f"  Min Samples Split: {bestParameters['RandomForest__min_samples_split']}",
        f"  Min Samples Leaf: {bestParameters['RandomForest__min_samples_leaf']}",
        "\nMigliori parametri per Logistic Regression:",
        f"  Penalty: {bestParameters['LogisticRegression__penalty']}",
        f"  C: {bestParameters['LogisticRegression__C']}",
        f"  Solver: {bestParameters['LogisticRegression__solver']}"
    ])

    # Salva il testo su file
    save_results_on_file(text)

def decision_tree_classifier(X_train, y_train, X_test, y_test, bestParameters):
    # Decision Tree Classifier
    dtc = DecisionTreeClassifier(
        criterion=bestParameters["DecisionTree__criterion"],
        max_depth=bestParameters["DecisionTree__max_depth"],
        min_samples_split=bestParameters["DecisionTree__min_samples_split"],
        min_samples_leaf=bestParameters["DecisionTree__min_samples_leaf"],
        random_state=42
    )

    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    dtc_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Prepara il testo da salvare
    text = "\n".join([
        "Random Forest Classifier:",
        f"  Accuracy: {accuracy:.4f}",
        f"  F1 Score: {f1:.4f}",
        f"  Precision: {precision:.4f}",
        f"  Recall: {recall:.4f}",
        f"  Balanced Accuracy: {dtc_accuracy:.4f}"
    ])

    # Salva il testo su file
    save_results_on_file(text)

def random_forest_classifier(X_train, y_train, X_test, y_test, bestParameters):
    # Random Forest Classifier
    rfc = RandomForestClassifier(
        criterion=bestParameters["RandomForest__criterion"],
        n_estimators=bestParameters["RandomForest__n_estimators"],
        max_depth=bestParameters["RandomForest__max_depth"],
        min_samples_split=bestParameters["RandomForest__min_samples_split"],
        min_samples_leaf=bestParameters["RandomForest__min_samples_leaf"],
        random_state=42
    )

    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    rfc_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Prepara il testo da salvare
    text = "\n".join([
        "Random Forest Classifier:",
        f"  Accuracy: {accuracy:.4f}",
        f"  F1 Score: {f1:.4f}",
        f"  Precision: {precision:.4f}",
        f"  Recall: {recall:.4f}",
        f"  Balanced Accuracy: {rfc_accuracy:.4f}"
    ])

    # Salva il testo su file
    save_results_on_file(text)

def logistic_regression(X_train, y_train, X_test, y_test, bestParameters):
    # Logistic Regression
    lr = LogisticRegression(
        penalty=bestParameters["LogisticRegression__penalty"],
        C=bestParameters["LogisticRegression__C"],
        solver=bestParameters["LogisticRegression__solver"],
        random_state=42
    )

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    lr_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Prepara il testo da salvare
    text = "\n".join([
        "Logistic Regression:",
        f"  Accuracy: {accuracy:.4f}",
        f"  F1 Score: {f1:.4f}",
        f"  Precision: {precision:.4f}",
        f"  Recall: {recall:.4f}",
        f"  Balanced Accuracy: {lr_accuracy:.4f}"
    ])

    # Salva il testo su file
    save_results_on_file(text)

def save_results_on_file(text):
    with open("results.txt", "a") as file:
        file.write("-------------------------------------------------\n")
        file.write(text + "\n")
        file.close()

def supervised(data, target):
    bestParameters, X_train, y_train, X_test, y_test = return_best_hyperparameters(data, target)
    
    save_best_hyperparameters(bestParameters)

    decision_tree_classifier(X_train, y_train, X_test, y_test, bestParameters)
    random_forest_classifier(X_train, y_train, X_test, y_test, bestParameters)
    logistic_regression(X_train, y_train, X_test, y_test, bestParameters)

    print("Risultati salvati su file.")

