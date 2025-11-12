
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


import utils.utils as utils
import utils.df_utils as dfu







def generate_sinusoidal_timeseries(n:int,
                                   f:int,
                                   freq_range:tuple[float,float]=(0.1, 1.0),
                                   amplitude_range:tuple[float,float]=(0.5, 2.0),
                                   phase_range:tuple[float,float]=(0, 2*np.pi),
                                   interaction_strength:float=0.1,
                                   seed:int|None=None,
                                   save_path:str|None=None,
                                  ) -> pd.DataFrame:
    '''
    Generate a DataFrame of shape (n, f) where each column is a sinusoid
    and features slightly interact with one another.

    **Arguments**:
    - `n` : Number of time-steps (samples).
    - `f` : Number of features.
    - `freq_range` : Min and max base frequency for each feature (in cycles per unit time).
    - `amplitude_range` : Min and max amplitude for each feature.
    - `phase_range` : Min and max phase (in radians) for each feature.
    - `interaction_strength` : Strength of coupling between features (0 means independent, larger means more coupling).
    - `seed` : Seed for reproducibility.
    - `save_path` : where to save the timeseries as a .csv file, if no path is provided then the dataframe is not saved

    **Returns**:
    - `df` : DataFrame with columns *“feat_0”*, *“feat_1”*, …, *“feat_{f-1}*”
    '''
    rng = np.random.default_rng(seed=seed)
    # time axis
    t = np.arange(n)

    # base parameters for each feature
    freqs = rng.uniform(freq_range[0], freq_range[1], size=f)
    amps = rng.uniform(amplitude_range[0], amplitude_range[1], size=f)
    phases = rng.uniform(phase_range[0], phase_range[1], size=f)

    # generate independent sinusoids
    X = np.zeros((n, f), dtype=float)
    for i in range(f):
        X[:, i] = amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])

    # add small cross‐feature interactions
    if interaction_strength > 0:
        # simple linear mixing of features: each feature gets a small addition
        # from the average of the other features
        other_mean = (X.sum(axis=1, keepdims=True) - X) / (f - 1)
        X = X + interaction_strength*other_mean

    # wrap in pandas DataFrame
    col_names = [f"feat_{i}" for i in range(f)]
    df = pd.DataFrame(X, columns=col_names)
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df



def add_label_to_timeseries(DF:pd.DataFrame, n_labels:int, label_feature:str, save_path:str|None=None) -> pd.DataFrame:
    '''
    Adds a 'label' feature for each dimension of the dataset.
    '''
    DF[f"label_{label_feature}"] = [min(n_labels-1,int((v+1)/2*n_labels)) for v in DF[label_feature]]
    if save_path is not None:
        DF.to_csv(save_path)
    return DF



def build_sequences(df:pd.DataFrame, seq_len:int, X_features:set[str]|None=None, Y_features:set[str]|None=None, verbose:bool=True) -> tuple[torch.Tensor, torch.Tensor]:
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
        Y_seq = Y[i + 1 : i + seq_len + 1]
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
        


def class_to_one_hot(idx:int, num_classes:int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Convert a single integer `idx` into a one-hot vector of size `num_classes`.
    """
    if idx < 0 or idx >= num_classes:
        raise ValueError(f"idx must be in [0, {num_classes-1}], got {idx}")
    # create a tensor with the index
    t = torch.tensor(idx, dtype=torch.long, device=device)
    one_hot = F.one_hot(t, num_classes=num_classes)  # uses torch.nn.functional.one_hot :contentReference[oaicite:0]{index=0}
    return one_hot.to(dtype)



def tensor_seq_to_one_hot(input_tensor:torch.Tensor, num_classes:int) -> torch.Tensor:
    """
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
                 verbose:bool=True
                ) -> pd.DataFrame:
    '''
    Loads the data from the ship-state simulations stored in `file_path` 
    '''
    if verbose:
        print(f"Loading data from '{file_path}' ...", end="")
    DF = dfu.get_dataframe(file_path=file_path)
    if verbose:
        print("done.")
    # Only get meaningful data
    if features is not None:
        if verbose:
            print("Extracting features ... ", end="")
        DF = DF[list(features)] if time_column in features else DF[list(features) + [time_column]]
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
        DF = (DF - DF.min()) / (DF.max() - DF.min())
        if verbose:
            print("done.")
    return DF






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
