import pickle
import time

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.metrics import correlation_score, log_likelihood_score
from pgmpy.models import BayesianNetwork
from sklearn.metrics import balanced_accuracy_score

from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 100)
label_encoder = LabelEncoder()

# Funzione che visualizza il grafo del Bayesian Network
def visualizeBayesianNetwork(bayesianNetwork: BayesianNetwork):
    G = nx.MultiDiGraph(bayesianNetwork.edges())
    pos = nx.spring_layout(G, iterations=100, k=2,
                           threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(G, pos, node_size=250, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=8,
        arrowstyle="->",
        edge_color="blue",
        connectionstyle="arc3,rad=0.2",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("BAYESIAN NETWORK GRAPH")
    plt.show()
    plt.clf()

def visualizeInfo(bayesianNetwork: BayesianNetwork):
    # Ottengo le distribuzioni di probabilità condizionata (CPD)
    for cpd in bayesianNetwork.get_cpds():
        print(f'CPD of {cpd.variable}:')
        print(cpd, '\n')

# Funzione che crea la rete bayesiana
def bNetCreation(df):
    # Ricerca della struttura ottimale
    hc_k2 = HillClimbSearch(df)
    k2_model = hc_k2.estimate(scoring_method='k2score', max_iter=100)
    # Creazione della rete bayesiana
    model = BayesianNetwork(k2_model.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    # Salvo la rete bayesiana su file
    with open('bn_model.pkl', 'wb') as output:
        pickle.dump(model, output)
    visualizeBayesianNetwork(model)
    return model

# Funzione che carica la rete bayesiana da file
def loadBayesianNetwork():
    with open('bn_model.pkl', 'rb') as input:
        model = pickle.load(input)
    visualizeBayesianNetwork(model)
    return model

# Predizione del valore di differentialColumn per l'esempio
def predict(bayesianNetwork: BayesianNetwork, example, differentialColumn):
    inference = VariableElimination(bayesianNetwork)
    result = inference.query(variables=[differentialColumn], evidence=example, elimination_order='MinFill')
    print(result)

# Generazione di un esempio randomico
def generateRandomExample(bayesianNetwork: BayesianNetwork):
    return bayesianNetwork.simulate(n_samples=1).drop(columns=['Target'])

# Funzione che esegue una query sulla rete bayesiana
def query_report(infer, variables, evidence=None, elimination_order="MinFill", show_progress=False, desc=""):
    if desc:
        print(desc)
    start_time = time.time()
    print(infer.query(variables=variables,
                      evidence=evidence,
                      elimination_order=elimination_order,
                      show_progress=show_progress))
    print(f'--- Query executed in {time.time() - start_time:0,.4f} seconds ---\n')

def create_load_bayesian_network(df):
    # Verifica se bn_model.pkl esiste allora carica la rete bayesiana altrimenti la crea
    try:
        with open('bn_model.pkl', 'rb') as input:
            print('Loading Bayesian Network from file...')
            bayesianNetwork = loadBayesianNetwork()
    except FileNotFoundError:
        print('Creating Bayesian Network...')
        bayesianNetwork = bNetCreation(df)

    return bayesianNetwork