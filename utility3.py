# -*- coding: utf-8 -*-
"""
Utility Functions

@author: eking
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.metrics import v_measure_score, completeness_score, homogeneity_score, adjusted_rand_score
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sys
from sklearn.decomposition import PCA, FastICA, randomized_svd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.backend import clear_session
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from scipy.stats import randint as sp_randint
from keras.layers import LeakyReLU
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score


print("funky chicken goes cluck cluck CLUCK!!!")
print("clucka clucka!!!")
print("coookoooocachoo!!!")
print('holy moley!')
print("bless you!!!")

def load_data(selector="motions", scale=True, valset=True, seed=42):
    
    if selector == 'motions':
        # Get into the right directory
        #print(os.getcwd())
        os.chdir("C:\\Users\\eking\\Google Drive\\1 - OMSA\\CS 7641 - Machine Learning\\datasets")
        #os.chdir("C:\\Users\\Erica\\Google Drive\\1 - OMSA\\CS 7641 - Machine Learning\\datasets")
        
        # Load Data
        raw_motions = pd.read_csv("motions.csv", header=None, delimiter=",")
        raw_motions.head()
        
        # Rename response into something sensical
        class_names=['rock', 'paper', 'scissors', 'okay']
        for i in range(4):
            raw_motions[64].replace(i, class_names[i], inplace=True)
        
        raw_motions.columns = [str(i) for i in raw_motions.columns]
        raw_motions.columns = raw_motions.columns.str.replace('64', 'motion_type')
        #raw_motions.head()
        
        # Split into response and predictors
        y = raw_motions[['motion_type']]
        X = raw_motions.drop(['motion_type'], axis=1)
        print("motions shapes: {}, {}".format(X.shape, y.shape))
        
    elif selector == 'particles':
        # Get into the right directory
        os.chdir("C:\\Users\\eking\\Google Drive\\1 - OMSA\\CS 7641 - Machine Learning\\datasets\\")
        #os.chdir("C:\\Users\\Erica\\Google Drive\\1 - OMSA\\CS 7641 - Machine Learning\\datasets")
        #print(os.getcwd())
        
        # Load Data
        df = pd.read_csv("smaller_particles.csv", header = 0, delimiter=",")
        df.head()
        
        # Set class names
        class_names=['proton', 'pion', 'kaon', 'positron']
        
        # Split into response and predictors
        y = df[['id']]
        X = df.drop(['id'], axis = 1)
        print("particles shapes: {}, {}".format(X.shape, y.shape))

    else:
        sys.exit("Please select either 'motions' or 'particles' datasets in load_data()")

    # Randomly divide into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    if valset:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    if scale:
        # Scale
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        if valset:
            X_val=scaler.transform(X_val)

    if valset:
        return X_train, X_val, X_test, y_train.values.ravel(), y_val.values.ravel(), y_test.values.ravel(), class_names
    else:
        return X_train, X_test, y_train.values.ravel(), y_test.values.ravel(), class_names
    
def plot_learning_curves(estimator, X_train, y_train, title = "Learning Curve", cv = 10, scorer = 'accuracy', low_limit = 0.8, ravel=True):
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    if ravel:
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train.values.ravel('C'), cv=cv, scoring = scorer)
    else:
        train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=cv, scoring=scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = "Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title = (title)
    plt.ylim(low_limit, 1)
    plt.legend(loc='best')
    plt.show()
    return 1


# Utility function to report best scores
def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
            
def plot_bars(scores_list_m, time_list_m, scores_list_p, time_list_p, test_parameter, n_range):
    n_groups = len(n_range)
    index = np.arange(n_groups)
    bar_width=0.35
    opacity=1
    
    plt.bar(index, scores_list_m, bar_width, 
                     alpha = opacity,
                     color = 'blue',
                     label = 'Motions')
    plt.bar(index + bar_width, scores_list_p, bar_width,
                     alpha = opacity,
                     color = 'green',
                     label = 'Particles')

    plt.xlabel(test_parameter)
    plt.ylabel('Validation set accuracy')
    plt.xticks(index + bar_width, n_range)
    plt.legend()
    plt.show()
    
    plt.bar(index, time_list_m, bar_width, 
                     alpha = opacity,
                     color = 'blue',
                     label = 'Motions')
    plt.bar(index + bar_width, time_list_p, bar_width,
                     alpha = opacity,
                     color = 'green',
                     label = 'Particles')
    
    plt.xlabel(test_parameter)
    plt.ylabel('Computation Time')
    plt.xticks(index + bar_width, n_range)
    plt.legend()
    plt.show()
    

def plot_lines(scores_list_m, time_list_m, scores_list_p, time_list_p, test_parameter, n_range):
    plt.plot(n_range, scores_list_m, color='blue', label='Motions')
    plt.plot(n_range, scores_list_p, color='green', label='Particles')
    plt.xlabel(test_parameter)
    plt.ylabel('Validation set accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(n_range, time_list_m, color='blue', label='Motions')
    plt.plot(n_range, time_list_p, color='green', label='Particles')
    plt.xlabel(test_parameter)
    plt.ylabel('Computation Time')
    plt.legend()
    plt.show()
    

def plot_lines1(scores_list, time_list, test_parameter, n_range, col = 'blue', label = 'Motions'):
    plt.plot(n_range, scores_list, color=col, label=label)
    plt.xlabel(test_parameter)
    plt.ylabel('Validation set loss')
    plt.legend()
    plt.show()
    
    plt.plot(n_range, time_list, color=col, label=label)
    plt.xlabel(test_parameter)
    plt.ylabel('Computation Time')
    plt.legend()
    plt.show()
    

#From https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", xlabel='x-var', ylabel='y-var', **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar



def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def make_plots(param_name_p4, param_range_p4, training_curves_p4, validation_curves_p4, time_list_p4):

    p4 = len(param_range_p4)
    curve_length = len(validation_curves_p4[0])
    print("hi~")
    color_idx4 = np.linspace(0, 1, p4)
    for i in range(p4):
        training_curve = training_curves_p4[i]
        validation_curve = validation_curves_p4[i]
        iter_label_p4 = param_range_p4[i]
        plt.plot(range(curve_length+1), validation_curve, color=plt.cm.winter(color_idx4[i]), label="{} = {}".format(param_name_p4, iter_label_p4))
        # plt.plot(range(curve_length+1), training_curve, color=plt.cm.summer(color_idx2[i]))#color=(i/len(param_range_p1)*0.9, 0, .8))#, label="max_attempts = {}".format(iter_label))
    plt.ylabel="validation log_loss"
    plt.xlabel="iterations"
    plt.legend()
    plt.show()

    val_last_p4 = []
    val_early_p4 = []
    val_mid1_p4 = []
    val_mid2_p4 = []
    for valcurve in validation_curves_p4:
        v = len(valcurve)
        val_last_p4.append(valcurve[v - 1])
        val_early_p4.append(valcurve[int(round(v*.1, 0))])
        val_mid1_p4.append(valcurve[int(round(v*.5,0))])
        val_mid2_p4.append(valcurve[int(round(v*.85))])

    n_lines = 4
    color_idx = np.linspace(0, 1, n_lines)

    plt.plot(param_range_p4, val_early_p4, color=plt.cm.cool(color_idx[0]), label="iteration 1000")
    plt.plot(param_range_p4, val_mid1_p4, color=plt.cm.cool(color_idx[1]), label="iteration 2000")
    plt.plot(param_range_p4, val_mid2_p4, color=plt.cm.cool(color_idx[2]), label="iteration 3500")
    plt.plot(param_range_p4, val_last_p4, color=plt.cm.cool(color_idx[3]), label="iteration 5000")
    plt.ylabel("validation log_loss")
    plt.xlabel(param_name_p4)
    plt.legend()
    plt.show()

    plt.plot(param_range_p4, time_list_p4, color='b', label="computation time as a function {}".format(param_name_p4))

def optimization_report(param_name, param_range, best_states, best_fitnesses, wine_times, save_filename=None):

    if save_filename!=None:
        save_pack = [param_name, param_range, best_states, best_fitnesses, wine_times]
        np.save(save_filename+".npy", save_pack)

    plt.plot(param_range, best_fitnesses)
    plt.show()
    plt.plot(param_range, wine_times)
    plt.show()

    best_index = best_fitnesses.index(min(best_fitnesses))
    print("-----Best Found-----")
    print("Best value for {}: {}".format(param_name, param_range[best_index]))
    print("Best GA state: {}".format(best_states[best_index]))
    print("Best GA fitness: {:.2f}".format(best_fitnesses[best_index]))
    print("GA Time @ Best: {:.2f}".format(wine_times[best_index]))

    return best_states[best_index], best_fitnesses[best_index]


def plot_weighted_graph(weighted_edges, node_color='purple', node_size=1000, edge_color="slategrey", font_color="white", font_size=16, scale_factor = 10):
    "Plot a weighted graph"

    fig = plt.figure()
    fig.set_facecolor("#FFFFFF")
    # 2. Add nodes
    G = nx.Graph()  # Create a graph object called G
    G.add_weighted_edges_from(weighted_edges)

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)

    node_list = G.nodes()
    # 3. If you want, add labels to the nodes
    nx.draw_networkx_labels(G, pos, font_color=font_color, font_size=font_size)

    all_weights = []
    # 4 a. Iterate through the graph nodes to gather all the weights
    for (node1, node2, edge_weight) in weighted_edges:
        all_weights.append(edge_weight)  # we'll use this when determining edge thickness

    # 4 b. Get unique weights and build a more pronounced scaling function
    unique_weights = list(set(all_weights))
    min_weight = min(unique_weights)
    max_weight = max(unique_weights)
    print("min_weight= {}".format(min_weight))
    print("max_weight= {}".format(max_weight))
    weight_diff = max_weight-min_weight

    # 4 c. Plot the edges - one by one!
    for weight in unique_weights:
        # 4 d. Form a filtered list with just the weight you want to draw
        the_nodes = [(node1, node2) for (node1, node2, weight) in weighted_edges]
        # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        width = (weight-min_weight)/weight_diff * scale_factor
        nx.draw_networkx_edges(G, pos, edgelist=the_nodes, width=width)

    # Plot the graph
    plt.axis('off')
    plt.show()

def plot_tso_graph(weighted_edges, best_state, color="darkorange"):

    G = nx.Graph()
    fig = plt.figure()

    for r in weighted_edges:
        u, v, length = r
        G.add_edge(u, v, length=length)

    m = len(best_state)
    selected_edges = [0]*m
    for i in range(m):
        selected_edges[i] = best_state[i % m], best_state[(i+1) % m]

    normal_edges = [(u, v) for (u, v, d) in weighted_edges if (u, v) not in selected_edges]

    dist_dict={}
    for (a, b, c) in weighted_edges:
        dist_dict[(a, b)] = c

    #set up positioning
    pos1 = nx.kamada_kawai_layout(G, dist=dist_dict)
    #draw nodes
    nx.draw_networkx_nodes(G, pos1, node_size=1000, node_color="midnightblue")
    #draw edges
    nx.draw_networkx_edges(G, pos1, edgelist=normal_edges, width=1, edge_color="cornflowerblue")
    nx.draw_networkx_edges(G, pos1, edgelist=selected_edges, width=6, edge_color=color, alpha=0.8, style="solid", arrows=True)

    nx.draw_networkx_labels(G, pos1, font_size=16, font_color='lightblue', font_weight="600")

    fig.set_facecolor("#ffffff")
    plt.axis('off')


def plot_mkcolor_graph(edge_data, best_state):
    fig = plt.figure()
    G = nx.Graph()
    G.add_edges_from(edge_data)
    G.add_nodes_from(range(len(best_state)))
    pos = nx.circular_layout(G)

    for i in range(len(best_state)):
            if best_state[i] == 1:
                nx.draw_networkx_nodes(G, pos, nodelist=[i],  node_color='blue', node_size=200)
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=[i], node_color='orange', node_size=200)
    nx.draw_networkx_edges(G, pos)#, edge_color="slategrey", width="3")

    fig.set_facecolor("#ffffff")
    plt.axis('off')
    plt.show()

def run_exp(param_name, param_range, model_type, other_params, metrics, bic=False):
    X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m, class_names_m = load_data('motions',
                                                                                          scale=True, valset=True)
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, class_names_p = load_data('particles',
                                                                                          scale=True, valset=True)

    result = defaultdict(list)
    for param in param_range:
        # print("testing number of clusters = {}".format(param))
        params = {param_name: param}
        # Motions
        t0 = time.time()

        if model_type=="gaussian mixture":
            cluster_m = GaussianMixture(**params, **other_params, covariance_type='full')
        elif model_type=="kmeans":
            cluster_m = KMeans(**params, **other_params)
        else:
            sys.exit("please select a valid model type, either 'gaussian mixture' or 'kmeans'")

        fit_m = cluster_m.fit(X_train_m)
        result['time_fit_m'].append(time.time() - t0)
        result['score_m'].append(cluster_m.score(X_train_m)/X_train_m.shape[1])
        result['score_val_m'].append(cluster_m.score(X_val_m)/X_val_m.shape[1])

        if bic and model_type=='gaussian mixture':
            result['bic_m'].append(cluster_m.bic(X_train_m))
            result['bic_val_m'].append(cluster_m.bic(X_val_m))

        t0 = time.time()
        y_pred_train_m = fit_m.predict(X_train_m)
        y_pred_val_m = fit_m.predict(X_val_m)
        result['time_pred_m'].append(time.time() - t0)

        for metric in metrics:
            result[metric+"_m"].append(getattr(sys.modules[__name__], metric)(y_train_m, y_pred_train_m))
            result[metric+'_val_m'].append(getattr(sys.modules[__name__], metric)(y_val_m, y_pred_val_m))

        # Particles
        t0 = time.time()
        if model_type=="gaussian mixture":
            cluster_p = GaussianMixture(**params, **other_params, covariance_type='tied')
        elif model_type=="kmeans":
            cluster_p = KMeans(**params, **other_params)
        else:
            sys.exit("please select either 'gaussian mixture' or 'kmeans'")

        fit_p = cluster_p.fit(X_train_p)
        result['time_fit_p'].append(time.time() - t0)
        result['score_p'].append(cluster_p.score(X_train_p)/X_train_p.shape[1])
        result['score_val_p'].append(cluster_p.score(X_val_p)/X_val_p.shape[1])

        if bic and model_type=='gaussian mixture':
            result['bic_p'].append(cluster_p.bic(X_train_p))
            result['bic_val_p'].append(cluster_p.bic(X_val_p))

        t0 = time.time()
        y_pred_train_p = fit_p.predict(X_train_p)
        y_pred_val_p = fit_p.predict(X_val_p)
        result['time_pred_p'].append(time.time() - t0)

        for metric in metrics:
            result[metric+"_p"].append(getattr(sys.modules[__name__], metric)(y_train_p, y_pred_train_p))
            result[metric+'_val_p'].append(getattr(sys.modules[__name__], metric)(y_val_p, y_pred_val_p))

    result['param_range'] = param_range
    result['param_name'] = param_name
    result['model_type'] = model_type
    result['metrics'] = metrics

    return result

def cluster_plots1(result, show_computation_time=True, log_likely_score=False):
    plot_result = result

    if show_computation_time:
        plt.plot(plot_result['param_range'], plot_result['time_pred_m'], color="blue", label="motions, pred_time",
                 linestyle=":")
        plt.plot(plot_result['param_range'], plot_result['time_pred_p'], color="orange", label="particles, pred_time",
                 linestyle=":")
        plt.plot(plot_result['param_range'], plot_result['time_fit_m'], color="blue", label="motions, fit_time",
                 linestyle="-")
        plt.plot(plot_result['param_range'], plot_result['time_fit_p'], color="orange", label="particles, fit_time",
                 linestyle="-")
        plt.ylabel("computation time")
        plt.xlabel(plot_result['param_name'])
        plt.legend()
        plt.show()

    if log_likely_score:
        plt.plot(plot_result['param_range'], plot_result['score_m'], color="blue", label="motions, train",
                 linestyle=":")
        plt.plot(plot_result['param_range'], plot_result['score_p'], color="orange", label="particles, train",
                 linestyle=":")
        plt.plot(plot_result['param_range'], plot_result['score_val_m'], color="blue", label="motions, val",
                 linestyle="-")
        plt.plot(plot_result['param_range'], plot_result['score_val_p'], color="orange", label="particles, val",
                 linestyle="-")
        plt.ylabel("normalized log_likelihood score")
        plt.xlabel(plot_result['param_name'])
        plt.legend()
        plt.show()

    for metric in result['metrics']:
        plt.plot(plot_result['param_range'], plot_result[metric+'_m'], color="blue", label="motions, train", linestyle=':')
        plt.plot(plot_result['param_range'], plot_result[metric+'_p'], color="red", label="particles, train", linestyle=':')
        plt.plot(plot_result['param_range'], plot_result[metric+'_val_m'], color="blue", label="motions, val", linestyle='-')
        plt.plot(plot_result['param_range'], plot_result[metric+'_val_p'], color="orange", label="particles, val", linestyle='-')
        plt.ylabel(metric)
        plt.xlabel(plot_result['param_name'])
        plt.legend()
        plt.show()


def decomp_exp(param_name, param_range, model_type, other_params, metrics):
    X_train_m, X_val_m, X_test_m, y_train_m, y_val_m, y_test_m, class_names_m = load_data('motions',
                                                                                          scale=True, valset=True)
    X_train_p, X_val_p, X_test_p, y_train_p, y_val_p, y_test_p, class_names_p = load_data('particles',
                                                                                          scale=True, valset=True)

    result = defaultdict(list)
    for param in param_range:
        # print("testing number of clusters = {}".format(param))
        params = {param_name: param}
        # Motions
        t0 = time.time()

        if model_type=="pca":
            cluster_m = PCA(**params, **other_params)
        elif model_type=="ica":
            cluster_m = FastICA(**params, **other_params)
        elif model_type == "rand_svd":
            cluster_m = randomized_svd(**params, **other_params)
        else:
            sys.exit("please select a valid model type, either 'pca' or 'ica', or 'rand_svd'")

        fit_m = cluster_m.fit(X_train_m)
        result['time_fit_m'].append(time.time() - t0)
        result['score_m'].append(cluster_m.score(X_train_m)/X_train_m.shape[1])
        result['score_val_m'].append(cluster_m.score(X_val_m)/X_val_m.shape[1])

        result['explained_variance_ratio'].append(cluster_m.explained_variance_ratio_)
        result['components'].append(cluster_m.components_)

        t0 = time.time()
        y_pred_train_m = fit_m.predict(X_train_m)
        y_pred_val_m = fit_m.predict(X_val_m)
        result['time_pred_m'].append(time.time() - t0)

        for metric in metrics:
            result[metric+"_m"].append(getattr(sys.modules[__name__], metric)(y_train_m, y_pred_train_m))
            result[metric+'_val_m'].append(getattr(sys.modules[__name__], metric)(y_val_m, y_pred_val_m))

        # Particles
        t0 = time.time()
        if model_type=="gaussian mixture":
            cluster_p = GaussianMixture(**params, **other_params)
        elif model_type=="kmeans":
            cluster_p = KMeans(**params, **other_params)
        else:
            sys.exit("please select either 'gaussian mixture' or 'kmeans'")

        fit_p = cluster_p.fit(X_train_p)
        result['time_fit_p'].append(time.time() - t0)
        result['score_p'].append(cluster_p.score(X_train_p)/X_train_p.shape[1])
        result['score_val_p'].append(cluster_p.score(X_val_p)/X_val_p.shape[1])

        result['explained_variance_ratio'].append(cluster_p.explained_variance_ratio_)
        result['components'].append(cluster_p.components_)

        t0 = time.time()
        y_pred_train_p = fit_p.predict(X_train_p)
        y_pred_val_p = fit_p.predict(X_val_p)
        result['time_pred_p'].append(time.time() - t0)

        for metric in metrics:
            result[metric+"_p"].append(getattr(sys.modules[__name__], metric)(y_train_p, y_pred_train_p))
            result[metric+'_val_p'].append(getattr(sys.modules[__name__], metric)(y_val_p, y_pred_val_p))

    result['param_range'] = param_range
    result['param_name'] = param_name
    result['model_type'] = model_type
    result['metrics'] = metrics

    return result

def cluster_plots1(result, show_computation_time=True, log_likely_score=False, bic=False, particles=True):
    plot_result = result

    if show_computation_time:
        plt.plot(plot_result['param_range'], plot_result['time_pred_m'], color="blue", label="motions, pred_time",
                 linestyle=":")
        plt.plot(plot_result['param_range'], plot_result['time_fit_m'], color="blue", label="motions, fit_time",
                 linestyle="-")
        if particles:
            plt.plot(plot_result['param_range'], plot_result['time_pred_p'], color="orange", label="particles, pred_time",
                     linestyle=":")
            plt.plot(plot_result['param_range'], plot_result['time_fit_p'], color="orange", label="particles, fit_time",
                     linestyle="-")
        plt.ylabel("computation time")
        plt.xlabel(plot_result['param_name'])
        plt.legend()
        plt.show()

    if log_likely_score:
        plt.plot(plot_result['param_range'], plot_result['score_m'], color="blue", label="motions, train",
                 linestyle=":")
        plt.plot(plot_result['param_range'], plot_result['score_val_m'], color="blue", label="motions, val",
                 linestyle="-")
        if particles:
            plt.plot(plot_result['param_range'], plot_result['score_p'], color="orange", label="particles, train",
                     linestyle=":")
            plt.plot(plot_result['param_range'], plot_result['score_val_p'], color="orange", label="particles, val",
                     linestyle="-")
        plt.ylabel("normalized log_likelihood score")
        plt.xlabel(plot_result['param_name'])
        plt.legend()
        plt.show()

    if bic:
        result['metrics'].append('bic')

    for metric in result['metrics']:
        plt.plot(plot_result['param_range'], plot_result[metric+'_m'], color="blue", label="motions, train", linestyle=':')
        plt.plot(plot_result['param_range'], plot_result[metric+'_val_m'], color="blue", label="motions, val", linestyle='-')
        if particles:
            plt.plot(plot_result['param_range'], plot_result[metric+'_p'], color="red", label="particles, train", linestyle=':')
            plt.plot(plot_result['param_range'], plot_result[metric+'_val_p'], color="orange", label="particles, val", linestyle='-')
        plt.ylabel(metric)
        plt.xlabel(plot_result['param_name'])
        plt.legend()
        plt.show()

def run_exp2(training, validation, param_name, param_range, model_type, other_params, metrics):
    X_train1_m, X_val1_m, X_test1_m, y_train_m, y_val_m, y_test_m, class_names_m = load_data('motions',
                                                                                          scale=True, valset=True)
    X_train1_p, X_val1_p, X_test1_p, y_train_p, y_val_p, y_test_p, class_names_p = load_data('particles',
                                                                                          scale=True, valset=True)
    [X_train_m, X_train_p] = training
    [X_val_m, X_val_p] = validation

    result = defaultdict(list)
    for param in param_range:
        # print("testing number of clusters = {}".format(param))
        params = {param_name: param}
        # Motions
        t0 = time.time()

        if model_type=="gaussian mixture":
            cluster_m = GaussianMixture(**params, **other_params, covariance_type='full')
        elif model_type=="kmeans":
            cluster_m = KMeans(**params, **other_params)
        else:
            sys.exit("please select a valid model type, either 'gaussian mixture' or 'kmeans'")

        fit_m = cluster_m.fit(X_train_m)
        result['time_fit_m'].append(time.time() - t0)
        result['score_m'].append(cluster_m.score(X_train_m)/X_train_m.shape[1])
        result['score_val_m'].append(cluster_m.score(X_val_m)/X_val_m.shape[1])

        t0 = time.time()
        y_pred_train_m = fit_m.predict(X_train_m)
        y_pred_val_m = fit_m.predict(X_val_m)
        result['time_pred_m'].append(time.time() - t0)

        for metric in metrics:
            result[metric+"_m"].append(getattr(sys.modules[__name__], metric)(y_train_m, y_pred_train_m))
            result[metric+'_val_m'].append(getattr(sys.modules[__name__], metric)(y_val_m, y_pred_val_m))

        # Particles
        t0 = time.time()
        if model_type=="gaussian mixture":
            cluster_p = GaussianMixture(**params, **other_params, covariance_type='tied')
        elif model_type=="kmeans":
            cluster_p = KMeans(**params, **other_params)
        else:
            sys.exit("please select either 'gaussian mixture' or 'kmeans'")

        fit_p = cluster_p.fit(X_train_p)
        result['time_fit_p'].append(time.time() - t0)
        result['score_p'].append(cluster_p.score(X_train_p)/X_train_p.shape[1])
        result['score_val_p'].append(cluster_p.score(X_val_p)/X_val_p.shape[1])

        t0 = time.time()
        y_pred_train_p = fit_p.predict(X_train_p)
        y_pred_val_p = fit_p.predict(X_val_p)
        result['time_pred_p'].append(time.time() - t0)

        for metric in metrics:
            result[metric+"_p"].append(getattr(sys.modules[__name__], metric)(y_train_p, y_pred_train_p))
            result[metric+'_val_p'].append(getattr(sys.modules[__name__], metric)(y_val_p, y_pred_val_p))

    result['param_range'] = param_range
    result['param_name'] = param_name
    result['model_type'] = model_type
    result['metrics'] = metrics

    return result



def run_exp_nn(X_train, y_train, X_val, y_val, param_name, param_range, other_params):

    result = defaultdict(list)

    '''
    ########## BEST FOUND PARAMETERS from HW1 #####
    n1 = 75
    n2 = 14
    mid_act = 'relu'  # useleakyrelu is enabled...
    num_layers = 3
    optimizer = 'adam'
    activation = 'sigmoid'
    epo = 100  # 10
    bat = 44  # 18
    ##############################################
    '''

    for param in param_range:
        clear_session()
        result['param'].append(param)
        params = {param_name: param}
        params.update(other_params)
        result['params'].append(params)
        result['metrics'].append('accuracy')
        # Motions
        t0 = time.time()

        num_features = X_train.shape[1]
        print('num_features = {}'.format(num_features))
        def classification_model(n1=75, n2=14, n3=14, num_layers=3,  input_dim=num_features,
                                 optimizer='adam', activation='sigmoid', epo=100, bat=44):
            model = Sequential()
            model.add(Dense(n1, input_dim=64))
            model.add(LeakyReLU())
            model.add(Dense(n2))
            model.add(LeakyReLU())
            for i in range(num_layers - 2):
                model.add(Dense(n3))
                model.add(LeakyReLU())
            model.add(Dense(4, activation=activation))
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            return model

        model = KerasClassifier(build_fn=classification_model, verbose=0, **params)

        model.fit(X_train, y_train.values.ravel('C'))

        y_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        result['accuracy_m'].append(accuracy_score(y_val, y_pred))
        result['accuracy_val_m'].append(accuracy_score(y_val, y_val_pred))
        print("took {} seconds".format(time.time() - t0))
        result['time'].append(time.time() - t0)

    # matplotlib is clunky in trying to plot bars side by side, BUT
    plot_lines1(result['param_range'], result['time'], result['param'], result['param_range'], label='Motions', col='blue')

    return result


