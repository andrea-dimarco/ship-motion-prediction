import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay


import utils




def get_avg_tree_impurity(dtree) -> float:
    n_leaves:int = 0
    tot_imp = 0.0
    
    n_nodes = dtree.tree_.node_count
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack` so we can loop through them
        if is_split_node:
            # is not leaf
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            # is leaf
            n_leaves += 1
            tot_imp += dtree.tree_.impurity[node_id] # entropy impurity at 'node'   
    return tot_imp/n_leaves # average impurity




def prune_sister_leaves(clf, parent_node_id):
    """
    Given a fitted *DecisionTreeClassifier* clf, collapse the two child-leaves
    of `parent_node_id` into a single leaf at that node.
    """
    from sklearn.tree import _tree
    tree = clf.tree_
    left_child  = tree.children_left[parent_node_id]
    right_child = tree.children_right[parent_node_id]

    # sanity checks
    if left_child < 0 or right_child < 0:
        raise ValueError(f"Node {parent_node_id} is already a leaf.")
    for child in (left_child, right_child):
        if (tree.children_left[child] != _tree.TREE_LEAF or
            tree.children_right[child] != _tree.TREE_LEAF):
            raise ValueError(f"Child node {child} is not a leaf.")

    # 1) Merge class‐count distributions:
    # tree.value[parent_node_id] = tree.value[left_child] + tree.value[right_child]

    # 2) Update sample counts (optional but good practice):
    # tree.n_node_samples[parent_node_id] = (
    #     tree.n_node_samples[left_child] + tree.n_node_samples[right_child]
    # )
    # if hasattr(tree, "weighted_n_node_samples"):
    #     tree.weighted_n_node_samples[parent_node_id] = (
    #         tree.weighted_n_node_samples[left_child]
    #         + tree.weighted_n_node_samples[right_child]
    #     )

    # 3) Turn parent into a leaf:
    tree.children_left[parent_node_id]  = _tree.TREE_LEAF
    tree.children_right[parent_node_id] = _tree.TREE_LEAF

    # 4) (Optional) reset impurity to zero if you want “pure” leaves:
    tree.impurity[parent_node_id] = 0.0

    clf.tree_ = tree
    return clf




def leaf_label(dtree, node_id):
    """
    Return the class label at the given leaf node_id.
    """
    # extract the raw counts: shape = (1, n_classes)
    counts = dtree.tree_.value[node_id][0]
    # pick the class with highest count
    class_index = np.argmax(counts)
    # map back to the actual class label
    return dtree.classes_[class_index]




def get_leaves(dtree, verbose:bool=False) -> set[tuple]:
    leaves:list = set()
    tot_imp = 0.0
    
    n_nodes = dtree.tree_.node_count
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, 0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth, parent = stack.pop()
        node_depth[node_id] = depth
        if verbose:
            print(f"Visiting node: id = {node_id}, left={children_left[node_id]}, right={children_right[node_id]}, parent={parent}")
        # If the left and right child of a node is not the same we have a split node
            # If a split node, append left and right children and depth to `stack` so we can loop through them
        if children_left[node_id] != children_right[node_id]:
            # is not leaf
            stack.append((children_left[node_id], depth + 1, node_id))
            stack.append((children_right[node_id], depth + 1, node_id))
        else:
            # is leaf
            leaves.add(node_id)
            tot_imp += dtree.tree_.impurity[node_id] # entropy impurity at 'node'   
    return leaves 




def get_split_nodes(dtree, verbose:bool=False) -> set[tuple]:
    split_nodes:list = set()
    tot_imp = 0.0
    
    n_nodes = dtree.tree_.node_count
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, 0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth, parent = stack.pop()
        node_depth[node_id] = depth
        if verbose:
            print(f"Visiting node: id = {node_id}, left={children_left[node_id]}, right={children_right[node_id]}, parent={parent}")
        # If the left and right child of a node is not the same we have a split node
            # If a split node, append left and right children and depth to `stack` so we can loop through them
        if children_left[node_id] != children_right[node_id]:
            # is not leaf
            stack.append((children_left[node_id], depth + 1, node_id))
            stack.append((children_right[node_id], depth + 1, node_id))
            split_nodes.add(node_id)
        else:
            # is leaf
            tot_imp += dtree.tree_.impurity[node_id] # entropy impurity at 'node'   
    return split_nodes 
        



def int_to_onehot(x:int, v_length:int) -> list[float]:
    onehot = [0.0 for _ in range(v_length)]
    onehot[x] = 1.0
    return onehot




def build_auc_plot(classifiers, X_data, y_data, file_path:str):
    fig, ax = plt.subplots()

    for classifier in classifiers:
        model_name = type(classifier).__name__
        y_pred = classifier.predict(X_data)
        y_pred = y_pred.argmax(axis=-1)
        fpr, tpr, _ = metrics.roc_curve(y_data,  y_pred)
        auc2 =  roc_auc_score(y_data, y_pred)
        plt.plot(fpr,tpr,label=f"{model_name}, auc={int(round(auc2*100))}%")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()
    plt.savefig(file_path, dpi=200)
    plt.clf()




def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")




def set_seed(seed=0) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False




