
import pandas as pd
import numpy as np

import utils.utils as utils



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




def generate_sinusoidal_timeseries_batch(n: int,
                                         f: int,
                                         batch_size: int = 1,
                                         freq_range=(0.1, 1.0),
                                         amplitude_range=(0.5, 2.0),
                                         phase_range=(0, 2*np.pi),
                                         interaction_strength=0.1,
                                         trend_slope_range=(0.0, 0.1),
                                         seasonal_amplitude_range=(0.0, 1.0),
                                         seasonal_period_range=(10, 100),
                                         noise_std_range=(0.0, 0.1),
                                         seed:int|None=None) -> pd.DataFrame:
    '''
    Generate a batch of synthetic multivariate time-series with:
      - `batch_size` independent series,
      - each series has `n` time-steps and `f` features,
      - each feature is a sinusoid + trend + seasonal component + noise,
      - plus minor interactions between features.

    Returns a pandas DataFrame of shape (`batch_size * n`, `f + metadata`) with a MultiIndex
    (series_id, time_index) or columns denoting series, etc.

    **Arguments**:
    - `n` : Number of time-steps per series.
    - `f` : Number of features per series.
    - `batch_size` : How many independent series to generate.
    - `freq_range` : Range (min, max) of base frequencies for the sinusoids (in cycles per series length).
    - `amplitude_range` : Range of amplitudes for the sinusoids.
    - `phase_range` : Range of starting phases (radians) for the sinusoids.
    - `interaction_strength` : Coupling strength between features (0 means independent).
    - `trend_slope_range` : Range of linear trend slopes per feature.
    - `seasonal_amplitude_range` : Range of amplitude for an additional seasonal (lower-frequency) sinusoid.
    - `seasonal_period_range` : Range of periods (in number of time-steps) for the seasonal component.
    - `noise_std_range` : Range of standard deviation for additive Gaussian noise per feature.
    - `seed` : Seed for reproducibility.

    **Returns**:
    - `df` : A DataFrame with MultiIndex (series_id, t) and columns feat_0 … feat_{f-1}.
    '''
    rng = np.random.default_rng(seed)
    # Create time axis 0..n-1
    t = np.arange(n)
    # Prepare storage: (batch_size, n, f)
    X_batch = np.zeros((batch_size, n, f), dtype=float)
    for b in range(batch_size):
        # draw base sinusoid params per feature
        freqs = rng.uniform(freq_range[0], freq_range[1], size=f)
        amps = rng.uniform(amplitude_range[0], amplitude_range[1], size=f)
        phases = rng.uniform(phase_range[0], phase_range[1], size=f)
        # trend params
        slopes = rng.uniform(trend_slope_range[0], trend_slope_range[1], size=f)
        # seasonal params (one extra slower sinusoid per feature)
        seasonal_amps = rng.uniform(seasonal_amplitude_range[0],
                                    seasonal_amplitude_range[1], size=f)
        seasonal_periods = rng.integers(seasonal_period_range[0],
                                         seasonal_period_range[1]+1,
                                         size=f)
        # noise std
        noise_stds = rng.uniform(noise_std_range[0], noise_std_range[1], size=f)
        # base X
        X = np.zeros((n, f), dtype=float)
        for i in range(f):
            # sinusoid
            X[:, i] = amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])
            # trend (linear)
            X[:, i] += slopes[i] * t
            # seasonal slow wave
            X[:, i] += seasonal_amps[i] * np.sin(2 * np.pi * t / seasonal_periods[i])
            # noise
            X[:, i] += rng.normal(loc=0.0, scale=noise_stds[i], size=n)
        # interactions: minor coupling via mean of other features
        if interaction_strength > 0 and f > 1:
            other_mean = (X.sum(axis=1, keepdims=True) - X) / (f - 1)
            X = X + interaction_strength * other_mean
        X_batch[b] = X
    # Now flatten into DataFrame
    # Create MultiIndex: series_id (0..batch_size-1), time (0..n-1)
    series_ids = np.repeat(np.arange(batch_size), n)
    times = np.tile(t, batch_size)
    multi_idx = pd.MultiIndex.from_arrays([series_ids, times],
                                          names=("series_id", "time_step"))
    # Flatten X_batch to shape (batch_size*n, f)
    X_flat = X_batch.reshape(batch_size * n, f)
    col_names = [f"feat_{i}" for i in range(f)]
    df = pd.DataFrame(X_flat, index=multi_idx, columns=col_names)
    return df




def add_label_to_timeseries(DF:pd.DataFrame, n_labels:int, save_path:str|None=None) -> pd.DataFrame:
    '''
    Adds a 'label' feature for each dimension of the dataset.
    '''
    for col in DF.columns:
        new_col_name = f"label_{col}"
        DF[new_col_name] = [min(n_labels-1,int((v+1)/2*n_labels)) for v in DF[col]]
    if save_path is not None:
        DF.to_csv(save_path)
    return DF



def generate_sequences(DF:pd.DataFrame, seq_len:int, save_path:str|None=None) -> pd.DataFrame:
    # TODO: this
    pass




if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
    
    # PARAMETERS
    color:str = "yellow"

    params:dict = utils.load_json("/data/params.json")


    DF = generate_sinusoidal_timeseries(n=100000,
                                        f=2,
                                        save_path="/data/dataset/data.csv",
                                        freq_range=(1.0, 1.1),
                                        amplitude_range=(0.1, 1.0),
                                        phase_range=(0.0, 1*np.pi),
                                        interaction_strength=0.1,
                                       )

    # from utils.plot_utils import plot_process
    # plot_process(DF.to_numpy(), labels=DF.columns, save_picture=True, show_plot=False, folder_path="/data/", title="test")

    DF = add_label_to_timeseries(DF=DF, n_labels=6, save_path="/data/dataset/data.csv")

    print(DF.head())
