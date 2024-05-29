# Import librerie utilizzate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import SMOTE

# Import delle funzioni da me definite
from plotting import *

def return_best_hyperparameters(dataset, target):
    # Cross Validation Strategy (Repeated Stratified K-Fold) with 12 splits and 2 repeats and a random state of 42 for reproducibility
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

    save_best_hyperparameters(bestParameters)

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

def train_model_k_fold(dataset, target, methodName, smote=False):
    bestParameters, X_train, y_train, X_test, y_test = return_best_hyperparameters(dataset, target)

    X = dataset.drop(target, axis=1).to_numpy()
    y = dataset[target].to_numpy()

    if smote:
        X, y = apply_smote(dataset, target)

    dtc = DecisionTreeClassifier(
        criterion=bestParameters["DecisionTree__criterion"],
        splitter="best",
        max_depth=bestParameters["DecisionTree__max_depth"],
        min_samples_split=bestParameters["DecisionTree__min_samples_split"],
        min_samples_leaf=bestParameters["DecisionTree__min_samples_leaf"],
    )
    rfc = RandomForestClassifier(
        n_estimators=bestParameters["RandomForest__n_estimators"],
        max_depth=bestParameters["RandomForest__max_depth"],
        min_samples_split=bestParameters["RandomForest__min_samples_split"],
        min_samples_leaf=bestParameters["RandomForest__min_samples_leaf"],
        criterion=bestParameters["RandomForest__criterion"],
        n_jobs=-1,
        random_state=42,
    )
    lr = LogisticRegression(
        penalty=bestParameters["LogisticRegression__penalty"],
        C=bestParameters["LogisticRegression__C"],
        solver=bestParameters["LogisticRegression__solver"],
        random_state=42,
    )

    cv = RepeatedStratifiedKFold(n_splits=sturgeRule(X_train.shape[0]), n_repeats=2, random_state=42)

    f1_scorer = make_scorer(f1_score, average="weighted")
    precision_scorer = make_scorer(precision_score, average="weighted")
    recall_scorer = make_scorer(recall_score, average="weighted")

    scoring_metrics = ["accuracy",f1_scorer,precision_scorer,recall_scorer,"balanced_accuracy"]

    for metric in scoring_metrics:
        # Cross Validation for each model with the scoring metric and the cross validation strategy
        scores_dtc = cross_val_score(
            dtc, X, y, scoring=metric, cv=cv, n_jobs=-1,
        )
        scores_rfc = cross_val_score(
            rfc, X, y, scoring=metric, cv=cv, n_jobs=-1
        )
        scores_lr = cross_val_score(
            lr, X, y, scoring=metric, cv=cv, n_jobs=-1
        )

        print("\033[94m")
        print(f"Metric: {metric}")
        print(f"DecisionTree: {scores_dtc.mean()}")
        print(f"RandomForest: {scores_rfc.mean()}")
        print(f"LogisticRegression: {scores_lr.mean()}")
        print("\033[0m")

        text = "\n".join([
        f"Metric: {metric}",
        f"DecisionTree: {scores_dtc.mean()}",
        f"RandomForest: {scores_rfc.mean()}",
        f"LogisticRegression: {scores_lr.mean()}",
        ])

        save_results_on_file(text)

    # Plotting the learning curves for each model
    plot_learning_curves(rfc, X, y, target, "RandomForest", methodName, cv)
    plot_learning_curves(dtc, X, y, target, "DecisionTree", methodName, cv)
    plot_learning_curves(lr, X, y, target, "LogisticRegression", methodName, cv)

def save_results_on_file(text):
    with open("results.txt", "a") as file:
        file.write("-------------------------------------------------\n")
        file.write(text + "\n")
        file.close()

# Applica la tecnica SMOTE per il bilanciamento delle classi
def apply_smote(dataset, target):
    X = dataset.drop(target, axis=1)
    y = dataset[target]

    smote = SMOTE(sampling_strategy='all', random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    return X_smote, y_smote
