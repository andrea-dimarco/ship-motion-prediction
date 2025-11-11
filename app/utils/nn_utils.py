import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.utils import data

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay


import utils.utils as utils
from typing import Any




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





class FFNetwork(nn.Module):
    def __init__(self,
                 data_dim:int,
                 hidden_dim:int,
                 out_dim:int=1,
                 num_layers:int=1,
                 bias:bool=True,
                ) -> None:
        '''
        The Feed Forward Regressor model.

        Arguments:
            - `data_dim`: dimension of the single sample 
            - `hidden_dim`: hidden dimension
            - `out_dim`: dimension of the output vector
            - `num_layers`: number of linear layers
        '''
        super().__init__()
        assert(data_dim > 0)
        assert(hidden_dim > 0)
        assert(num_layers > 0)

        self.data_dim = data_dim
        # Initialize Modules
        # input = ( batch_size, lookback*data_dim )
        model = [
            nn.Linear(in_features=data_dim, out_features=hidden_dim, bias=bias),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        ]
        for _ in range(num_layers-1):
            model += [
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=bias),
                nn.ReLU(inplace=True),
            ]
        model += [
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        ]
        self.feed = nn.Sequential(*model)
        # init weights
        self.feed.apply(init_weights)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, data_dim]

        Returns:
            - the predicted sequences [batch, data_dim]
        '''
        # x = (batch, data)
        x = self.feed(x)
        # x = ( batch, out_dim )
        return x




class FFClassifier(nn.Module):
    def __init__(self,
                 data_dim:int,
                 hidden_dim:int,
                 out_dim:int,
                 num_layers:int=1,
                 bias:bool=True,
                ) -> None:
        '''
        The Feed Forward Classifier model.

        Arguments:
            - `data_dim`: dimension of the single sample 
            - `hidden_dim`: hidden dimension
            - `out_dim`: dimension of the output vector
            - `num_layers`: number of linear layers
        '''
        super().__init__()
        assert(data_dim > 0)
        assert(hidden_dim > 0)
        assert(num_layers > 0)

        self.data_dim = data_dim
        # Initialize Modules
        # input = ( batch_size, lookback*data_dim )
        model = [
            nn.Linear(in_features=data_dim, out_features=hidden_dim, bias=bias),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        ]
        for _ in range(num_layers-1):
            model += [
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=bias),
                nn.ReLU(inplace=True),
            ]
        model += [
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
        ]
        self.softmax = nn.Softmax(dim=1) # dim means over which dimension the softmax is to be performed
        self.feed = nn.Sequential(*model)
        # init weights
        self.feed.apply(init_weights)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Takes the noise and generates a batch of sequences

        Arguments:
            - `x`: input of the forward pass with shape [batch, data_dim]

        Returns:
            - the predicted sequences [batch, data_dim]
        '''
        # x = (batch, data)
        x = self.feed(x)
        # x = ( batch, out_dim )
        x = self.softmax(x)
        return x




class PositionalEncoder(nn.Module):
    def __init__(self, sample_length, max_seq_len=5000):
        super(PositionalEncoder, self).__init__()
        
        # Create a long positional encoding matrix (max_seq_len x sample_length)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, sample_length, 2).float() * (-math.log(10000.0) / sample_length))
        
        pe = torch.zeros(max_seq_len, sample_length)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # Non-trainable

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, sequence_length, sample_length)
        Returns:
            Tensor of same shape as x, with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)  # (1, seq_len, sample_length)
        return x
    



class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src):
        return self.layers(src)




class TransformerClassifier(nn.Module):
    # TODO: read this https://jamesmccaffrey.wordpress.com/2023/04/10/example-of-a-pytorch-multi-class-classifier-using-a-transformer/
    # TODO: read this https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
    def __init__(self, input_dim:int, num_classes:int, vocab_size:int=None, d_model:int=512,
                 nhead:int=8, num_layers:int=1, dim_feedforward:int=2048,
                 dropout:float=0.1, max_len:int=5000, pad_idx:int=0, use_softmax:bool=True,
                 name:str="TransformerClassifier"
                ):
        super().__init__()
        self.use_softmax = use_softmax
        self.name = name
        self.d_model = d_model
        self.pad_idx = pad_idx

        if vocab_size is None:
            self.linear = nn.Linear(input_dim, d_model)
            self.use_embedding = False
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.use_embedding = True
        self.pos_encoder = PositionalEncoder(d_model, max_len)

        self.encoder = TransformerEncoder(num_layers=num_layers,
                                          d_model=d_model,
                                          nhead=nhead,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                         )

        self.classifier = nn.Linear(d_model, num_classes)
        if use_softmax:
            self.softmax = nn.Softmax(dim=1)


    def make_pad_mask(self, seq):
        # seq: (batch_size, seq_len, sample_size)
        return (seq == self.pad_idx)

    def forward(self, src:torch.Tensor):
        """
        - **src** has size (`batch_size`, `seq_len`, `sample_size`)
        - **return** tensor has size (`batch_size`, `num_classes`)
        """
        # Embed + positional encoding
        if self.use_embedding:
            x = self.embedding(src) * math.sqrt(self.d_model)
        else:
            x = self.linear(src) # (batch, seq_len, d_model)
        x = self.pos_encoder(x) # (batch, seq_len, d_model)
        # Padding mask
        padding_mask = self.make_pad_mask(x) # (batch, seq, d_model)
        x = torch.einsum('xyz,xyz->xyz',[x,padding_mask]) # apply mask
        x = self.encoder(x) # (batch_size, seq_len, d_model) <-- encoded
        # Pooling: Take the first token or average
            # can be swapped with max pooling.

        x = x.mean(dim=1) # (batch_size, d_model)
        x = self.classifier(x)  # (batch_size, num_classes)

        if self.use_softmax:
            return self.softmax(x)
        else:
            return x
        



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




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def init_weights(m):
    '''
    Initialized the weights of the nn.Sequential block
    '''
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.zeros_(m.bias)



def train_model(model,
                X_train:torch.Tensor,
                y_train:torch.Tensor,
                X_val:torch.Tensor,
                y_val:torch.Tensor,
                n_epochs:int,
                batch_size:int,
                device:Any|None=None,
                adam_lr:float=0.01,
                adam_b1:float=0.75,
                adam_b2:float=0.90,
                decay_start:float=1.0,
                decay_end:float=1.0,
                loss_plot_folder:str|None=None,
                model_name:str="DeepModel",
                loss_fn=nn.CrossEntropyLoss(),
                val_frequency:int=100,
                save_folder:str="/data/models/",
                verbose:bool=True,
                color:str="blue",
               ):
    '''
    Instanciate, train and return the model the model.

    Arguments:
        - `X_train`: train Tensor [n_sequences, data_dim]
        - `y_train`: the targets
        - `X_val`: seques for validation
        - `y_val`: targets for validation
        - `plot_loss`: if to plot the loss or not
        - `loss_fn`: the loss function to use
        - `val_frequency`: after how many epochs to run a validation epoch
    '''
    input_size:int = X_train.size()[1]
    if device is None:
        device = get_device()
    if verbose:
        print(f"Using device", end=" ")
        utils.print_colored(device, color=color, end=".\n")

    if verbose:
        utils.print_colored(model_name, color=color, end="")
        print(" model has ", end="")
        utils.print_colored(count_parameters(model), color=color, end="")
        print(" parameters.")

    optimizer = optim.Adam(model.parameters(), lr=adam_lr, betas=(adam_b1, adam_b2))
    lr_scheduler = optim.lr_scheduler.LinearLR(optimizer,
                                               start_factor=decay_start,
                                               end_factor=decay_end,
                                               total_iters=n_epochs
                                              )
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                                   shuffle=True,
                                   batch_size=batch_size
                                  )

    if loss_plot_folder is not None:
        loss_history = []
    if verbose:
        print("Training started.")
    if verbose:
        timer = utils.TimeExecution()
        timer.start()
        full_timer = utils.TimeExecution()
        full_timer.start()
    for epoch in range(n_epochs):
        # TRAINING STEP
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch.to(device=device))
            loss = loss_fn(y_pred, y_batch.to(device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # VALIDATION STEP: every `val_frequency` epochs
        if (epoch % val_frequency) == 0:
            model.eval()
            with torch.no_grad():
                # loss on TRAINING set
                y_pred = model(X_train)
                train_loss = loss_fn(y_pred, y_train)
                # loss on VALIDATION set
                y_pred = model(X_val)
                val_loss = loss_fn(y_pred, y_val)
                if loss_plot_folder is not None:
                    loss_history.append(val_loss.item())
            if verbose:
                # STATUS MESSAGE
                timer.end()
                print(f"Epoch ", end=" ")
                utils.print_colored(f"{epoch}/{n_epochs}", highlight=color, end=": ")
                print("train_loss=", end="")
                utils.print_colored(round(float(train_loss),5), color=color, end="; ")
                print("val_loss=", end="")
                utils.print_colored(round(float(val_loss),5), color=color, end="; ")
                print("lr=", end="")
                utils.print_colored(optimizer.param_groups[0]['lr'], color=color, end="; ")
                print("elapsed_time=", end="")
                utils.print_colored(str(timer), color=color)
                timer.start()
        lr_scheduler.step()
    if verbose:
        full_timer.end()
        full_timer.print()
    # PLOT LOSS
    if loss_plot_folder is not None:
        plt.plot(loss_history, label="validation loss")
        plt.plot([min(loss_history[:i]) for i in range(1,len(loss_history)+1)], label="Minimum Loss")
        plt.grid()
        plt.legend()
        plt.savefig(f"{loss_plot_folder}{model_name}-loss.png")
    # SAVE TRAINED MODEL 
    if save_folder is not None:
        torch.save(model.state_dict(), f"{save_folder}{model_name}.pth")
    return model




def fix_class_imbalance(DF:pd.DataFrame, target_column:str, downsample:bool=True, verbose:bool=True) -> pd.DataFrame:
    labels:set = sorted(list(set(DF[target_column])))
    label_count:dict[int|str, int] = dict()
    new_DF:pd.DataFrame = None
    for l in labels:
        label_count[l] = len(DF[DF[target_column] == l])
    # FIX CLASS IMBALANCE
    if downsample:
        if verbose:
            print("Fixing class imbalance with ", end="")
            utils.print_colored("down", color="red", end="-sampling ... ")
        min_count:int = min(label_count.values())
        for l in label_count.keys():
            remove_idx:list[int] = list()
            current_count:int = label_count[l]
            for idx, row in DF.iterrows():
                # remove random samples
                if row[target_column] == l:
                    remove_idx.append(idx)
                    current_count -= 1
                    if current_count <= min_count:
                        break
        new_DF = DF.drop(remove_idx)
        new_DF.reset_index(inplace=True)
        new_DF.drop(columns=['index'], inplace=True)
        if verbose:
            print("done.")
    else:
        # UPSAMPLE
        if verbose:
            print("Fixing class imbalance with ", end="")
            utils.print_colored("up", color="red", end="-sampling ... ")
        max_count:int = max(label_count.values())
        for l in label_count.keys():
            if max_count - label_count[l] > 0:
                labeled_samples = DF[DF[target_column] == l] # get samples with current label
                new_rows = labeled_samples.sample(n=max_count-label_count[l], replace=True) # get random rows
                new_DF = utils.concat_dfs(new_DF, new_rows)
        new_DF.reset_index(inplace=True)
        new_DF.drop(columns=['index'], inplace=True)
        if verbose:
            print("done.")
    return new_DF




class REPORT:

    @classmethod
    def print_report(name, report):
        cv = len(list(report.values())[0]['cross_val'])

        rows = [
                [ '-', 'precision', 'recall', 'f1-score', 'support', 'roc', 'confusion matrix', f'cross val (cv={cv})' ],
                [ '(class)', ['0', '1'], ['0', '1'], ['0', '1'], ['0', '1'], '-', '-', '-' ],
               ]

        for model_name, data in report.items():
            metrics = [ [ data['metrics'][c][metric] for c in [ '0', '1' ] ] for metric in [ 'precision', 'recall', 'f1-score', 'support' ] ]
            row = [model_name, *metrics, data['roc'], REPORT.render_confusion_matrix(data['confusion_matrix']), REPORT.render_cross_val(data['cross_val'])]
            rows.append(row)

        print(REPORT.build_table(rows, heading=name))

    @classmethod
    def build_table(rows, heading=None):
        if not rows:
            return ""

        rows = list(map(lambda x: REPORT.build_row(x), rows))
        row_separator = "-" * len(rows[0])
        output = row_separator + "\n"

        if heading:
            output += REPORT.build_row([ heading ], width=len(rows[0]) - 2) + "\n" + row_separator + "\n"

        for row in rows:
            output += row + "\n" + row_separator + "\n"

        return output

    @classmethod
    def render_cross_val(cv):
        return f"{cv.mean()*100:.2f}% ({cv.std():.2f})"
    
    @classmethod
    def render_confusion_matrix(cf):
        return f"{cf[0][0]} {cf[0][1]} - {cf[1][0]} {cf[1][1]}"
    
    @classmethod
    def build_row(components, width=31):
        center = "|".join([REPORT.build_cell(component, width=width) for component in components])
        row = f"|{center}|"
        return row
    
    @classmethod
    def build_cell(component, width=31):
        if type(component) == str:
            return component.center(width)
        elif type(component) == int:
            return f"{component}".center(width)
        elif type(component) == np.float64 or type(component) == float:
            return f"{component*100:.2f}%".center(width)
        elif type(component) == list or type(component) == set:
            return "|".join(map(lambda x: REPORT.build_cell(x, width=width//len(component)), component))
        else:
            print(component)
            print(type(component))

    @classmethod
    def create_report(X_data, y_data, classifiers, cv=5):
        report = {}

        for model_name, classifier in classifiers:
            y_pred = classifier.predict(X_data)
            y_pred = y_pred.argmax(axis=-1)

            # estimator = KerasClassifier(build_fn= lambda : classifier, batch_size = 1, epochs = 100)
            report[model_name] = {}
            report[model_name]['roc'] = roc_auc_score(y_data, y_pred)
            report[model_name]['confusion_matrix'] = confusion_matrix(y_data, y_pred)
            #cambiamento di X, Y in features, label
            report[model_name]['cross_val'] = np.array([]) #cross_val_score(estimator, features, label, cv=cv)
            report[model_name]['metrics'] = REPORT.classification_report(y_data, y_pred, output_dict=True)

        return report
    
    @classmethod
    def show_details(X_data, y_data, model, cv):
        model_name, classifier = model
        y_pred = classifier.predict(X_data)
        y_pred = y_pred.argmax(axis=-1)
        #cambiamento di X, Y in features, label
        # estimator = KerasClassifier(build_fn= lambda : classifier, batch_size = 1, epochs = 100)
        # scores = cross_val_score(estimator, features, label, cv=cv)
        class_report = REPORT.classification_report(y_data, y_pred)
        confusion = confusion_matrix(y_data, y_pred)
        roc = roc_auc_score(y_data, y_pred)
        _, accuracy = classifier.evaluate(X_data, y_data, verbose=1)

        print(f" {model_name} ".center(60, "="))

        print(class_report)
        print(confusion)
        print(f"{model_name} roc: {roc*100:.2f}%")
        # print(scores)
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        print(f"{accuracy * 100}% accuracy")

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["0", "1"])
        disp.plot(cmap=plt.cm.Blues)
        disp.ax_.set_title(model_name)
        plt.show()

        build_auc_plot([classifier], X_data, y_data)

        print("\n\n")




