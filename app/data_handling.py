
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from typing import Literal

import utils.utils as utils
import utils.df_utils as dfu



def get_data(task:Literal['REGR','CLAS'],
             seq_len:int,
             dataset_file:str,
             in_features:set[str],
             out_features:set[str]|None=None,
             n_labels:int|None=None,
             verbose:bool=True,
             reduce_frequency:bool=False,
            ) -> tuple[torch.Tensor,torch.Tensor]:
    from numpy import pi
    if out_features is None:
        out_features = in_features.copy()
    elif in_features is None:
        in_features = out_features.copy()
    DF = load_dataset(file_path=dataset_file,
                      features=in_features.union(out_features),
                      verbose=verbose,
                      normalize=True,
                      reduce_frequency=reduce_frequency,
                     )
    DF = DF[sorted(list(in_features.union(out_features)))]
    if task == 'REGR':
        X, Y = build_sequences(df=DF,
                               X_features=sorted(in_features),
                               Y_features=sorted(out_features),
                               seq_len=seq_len,
                               verbose=verbose
                              )
    elif task == 'CLAS':
        assert len(out_features) == 1, "Multi-label samples are not supported (yet)"
        label_feature = list(out_features)[0]
        DF = add_label_to_timeseries(DF, n_labels=n_labels, label_feature=label_feature)
        X, Y = build_sequences_labels(df=DF,
                                      seq_len=seq_len,
                                      labels_columns=[f'label_{label_feature}'],
                                      verbose=verbose
                                     )
        Y = tensor_seq_to_one_hot(Y, num_classes=n_labels) # NOTE: sequences are UNIVARIATE, must be updated for MULTIVARIATE SEQUENCES
    else:
        raise ValueError(f"Unsupported task ({task})")
    # only get last element of sequence for training
    Y = Y[:, -1, :]  # shape (n_sequences, sample_size)
    return X, Y
    

def label_to_range(label, n_labels:int, minimum:float=-1.0, maximum:float=1.0) -> tuple[float,float]:
    range = maximum - minimum
    step = range/n_labels
    return (label)*step+minimum, (label+1)*step+minimum


def add_label_to_timeseries(DF:pd.DataFrame, n_labels:int, label_feature:str, save_path:str|None=None) -> pd.DataFrame:
    '''
    Adds a 'label' feature for each dimension of the dataset. Values **must** already be in [-1,+1]
    '''
    DF[f"label_{label_feature}"] = [min(n_labels-1, int((v+1)/2*n_labels)) for v in DF[label_feature]]
    if save_path is not None:
        DF.to_csv(save_path)
    return DF



def build_sequences(df:pd.DataFrame,
                    seq_len:int,
                    X_features:set[str]|None=None,
                    Y_features:set[str]|None=None,
                    verbose:bool=True,
                    look_ahead:int=1, # TODO: test look_ahead higher than 1
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build input (X) and target (Y) tensors from a pandas DataFrame for sequence modeling.
    
    **Arguments**:
    - `df` : The input DataFrame containing time-ordered data.
    - `seq_len` : he length of each input sequence.
    
    **Returns**:
    - `X` : Tensor of shape (num_sequences, seq_len, num_features)
    - `Y` : Tensor of shape (num_sequences, seq_len, num_features)
        Same as X but shifted one timestep ahead.
    """
    assert look_ahead > 0
    X = (df[list(X_features) if X_features is not None else df]).values.astype('float32')
    Y = (df[list(Y_features) if Y_features is not None else df]).values.astype('float32')
    assert len(X) == len(Y)
    num_sequences = len(X) - seq_len
    X_list, Y_list = [], []
    if verbose:
        print("Composing sequences")
        bar = utils.BAR(num_sequences)
    for i in range(num_sequences):
        X_seq = X[i : i + seq_len]
        Y_seq = Y[i + look_ahead : i + seq_len + look_ahead]
        X_list.append(X_seq)
        Y_list.append(Y_seq)
        if verbose:
            bar.update()
    X = torch.tensor(X_list)
    Y = torch.tensor(Y_list)
    return X, Y



def build_sequences_labels(df:pd.DataFrame, seq_len:int, labels_columns:list[str], verbose:bool=True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build input (X) and target (Y) tensors from a pandas DataFrame for sequence modeling.
    
    **Arguments**:
    - `df` : The input DataFrame containing time-ordered data.
    - `seq_len` : he length of each input sequence.
    
    **Returns**:
    - `X` : Timeseries
    - `Y` : Timeseries of discrete values (labels)
    """
    non_label_columns = list(set(df.columns) - set(labels_columns))
    num_sequences = len(df) - seq_len
    NON_LABELS = df[non_label_columns].values.astype('float32')
    LABELS = df[labels_columns].values.astype('float32')
    X_list, Y_list = [], []
    if verbose:
        print("Composing sequences")
        bar = utils.BAR(num_sequences)
    for i in range(num_sequences):
        X_seq = NON_LABELS[i : i + seq_len]
        Y_seq = LABELS[i + 1 : i + seq_len + 1]
        X_list.append(X_seq)
        Y_list.append(Y_seq)
        if verbose:
            bar.update()
    X = torch.tensor(X_list)
    Y = torch.tensor(Y_list)
    return X, Y
        


def class_to_onehot(idx:int, num_classes:int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Convert a single integer `idx` into a one-hot vector of size `num_classes`.
    """
    if idx < 0 or idx >= num_classes:
        raise ValueError(f"idx must be in [0, {num_classes-1}], got {idx}")
    # create a tensor with the index
    t = torch.tensor(idx, dtype=torch.long, device=device)
    one_hot = F.one_hot(t, num_classes=num_classes)  # uses torch.nn.functional.one_hot :contentReference[oaicite:0]{index=0}
    return one_hot.to(dtype)



def onehot_to_class(vector:torch.Tensor) -> int:
    """
    onverts a one-hot encoded class or a probability distribution to the corresponding class
    """
    return torch.argmax(vector).item()




def tensor_seq_to_one_hot(input_tensor:torch.Tensor, num_classes:int) -> torch.Tensor:
    """
    **SEQUENCES MUST BE UNNIVARIATE**.
    
    Convert a tensor of shape (n_sequences, seq_len, 1) where each entry is an integer in [0, num_classes-1]
    into a one-hot encoded tensor of shape (n_sequences, seq_len, num_classes).
    """
    # check shape
    if input_tensor.ndim != 3 or input_tensor.size(-1) != 1:
        raise ValueError(f"input_tensor must have shape (n_sequences, seq_len, 1); got {input_tensor.shape}")
    # flatten the last dim
    indices = input_tensor.squeeze(-1).long()  # shape becomes (n_sequences, seq_len)
    # do one_hot: result shape (n_sequences, seq_len, num_classes)
    one_hot = F.one_hot(indices, num_classes=num_classes)
    # optionally convert type to float if you plan to feed into a network
    return one_hot.float()



def load_dataset(file_path:str,
                 features:set|None=None,
                 time_column:str='time',
                 normalize:bool=True,
                 verbose:bool=True,
                 reduce_frequency:bool=False,
                ) -> pd.DataFrame:
    '''
    Loads the data from the ship-state simulations stored in `file_path` 
    '''
    if verbose:
        print(f"Loading data from '{file_path}' ... ", end="")
    DF = dfu.get_dataframe(file_path=file_path)
    if verbose:
        print("done.")
    # Only get meaningful data
    if features is not None:
        if verbose:
            print("Extracting features ... ", end="")
        DF = DF[sorted(list(features))] if time_column in features else DF[sorted(list(features) + [time_column])]
        if verbose:
            print("done.")
    # first row is the unit of measurement
    DF = DF[1:].astype(float)
    if verbose:
        print("Sorting samples ... ", end="")
    DF.sort_values([time_column], inplace=True)
    if verbose:
        print("done.")
    if time_column not in features:
        DF.drop(columns=[time_column], inplace=True)
    if normalize:
        if verbose:
            print("Normalizing data (min-max) ... ", end="")
        DF = ((DF - DF.min()) / (DF.max() - DF.min()))*2 -1
        if verbose:
            print("done.")
    if reduce_frequency:
        if verbose:
            print("Reducing dataset frequency (half) ... ", end="")
        DF = dfu.get_even_rows(DF)
        if verbose:
            print("done.")
    return DF




def train_test_split(X:torch.Tensor, y:torch.Tensor, split:float=0.75) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Splits the two tensors `X` and `y` in two at the same cutting point defined by `split`
    '''
    assert split > 0.0 and split < 1.0
    split_index = int(X.shape[0] * split)
    
    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test







if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    # PARAMETERS
    color:str = "yellow"

    params:dict = utils.load_json("/data/params.json")


    DF = load_dataset(file_path=f"{params['dataset_folder']}/{params['dataset']}",
                      features=set(params['input_features']),
                     )
    
    import utils.plot_utils as pl
    pl.plot_processes(samples=DF.to_numpy(),
                      labels=params['input_features'],
                      folder_path=params['img_folder'],
                      title=params['dataset'],
                      img_name=params['dataset'],
                     )
