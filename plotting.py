import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Definizione della funzione sturgeRule
def sturgeRule(n):
    return int(1 + 3.322 * np.log10(n))

# Funzione che mostra la curva di apprendimento per ogni modello
def plot_learning_curves(model, X, y, differentialColumn, model_name, method_name, cv, scoring='balanced_accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, random_state=42)

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Salva su file i valori numerici della deviazione standard e della varianza
    print(
        f"\033[94m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")
    f = open(f'plots/var_std_{model_name}_{method_name}.txt', 'w')
    f.write(f"Train Error Std: {train_errors_std[-1]}\n")
    f.write(f"Test Error Std: {test_errors_std[-1]}\n")
    f.write(f"Train Error Var: {train_errors_var[-1]}\n")
    f.write(f"Test Error Var: {test_errors_var[-1]}\n")
    f.close()

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    # Visualizza la curva di apprendimento
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name} con {method_name}')
    # plt.ylim(0, 1)
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()

    # save plot to file
    plt.savefig(f'plots/learning_curve_{model_name}_{method_name}.png')

    plt.show()