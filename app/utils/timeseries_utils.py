# project libraries
import utils.utils as utils

# internal libraries
import datetime

# external libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.api import OLS





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





def save_timeseries(samples, folder_path:str, file_name="timeseries.csv") -> None:
    '''
    Save the samples as a csv file.
    '''
    # Save it
    df = pd.DataFrame(samples)
    df.to_csv(f"{folder_path}{file_name}", index=False, header=False)



def plot_ACF_PACF(timeseries:np.ndarray,
                  output_folder:str,
                  series_name:str,
                  max_lag:int=10,
                  verbose:bool=False
                 ) -> None:
    '''
    Plots ACF and PACF plots
    '''
    if verbose:
        print("Generating diagnostic check plots ... ", end="")
    lags = min(max_lag, len(timeseries))
    # Plot the ACF of the residuals
    plt.figure(figsize=(7, 6))
    sm.graphics.tsa.plot_acf(timeseries, lags=lags)
    plt.title(f"'{series_name}' ACF Plot")
    plt.grid()
    plt.savefig(f"{output_folder}{series_name.replace('/', '-')}-ACF.png")
    plt.clf()
    # Plot the PACF of the residuals
    plt.figure(figsize=(7, 6))
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags)
    plt.title(f"'{series_name}' PACF Plot")
    plt.grid()
    plt.savefig(f"{output_folder}{series_name.replace('/', '-')}-PACF.png")
    plt.clf()
    if verbose:
        print("done.")



def diagnostic_check(model, output_folder:str, lag=10, model_name:str="", verbose:bool=False) -> None:
    '''
    Check timeseries model performance
    '''
    if verbose:
        print("Generating residual diagnostic check plots ... ", end="")
    # Plot the residuals
    residuals = model.resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title(f'Residuals of the {model_name} Model')
    plt.savefig(f"{output_folder}residuals.png")
    plt.clf()

    # Plot the ACF of the residuals
    plt.figure(figsize=(10, 6))
    sm.graphics.tsa.plot_acf(residuals, lags=lag)
    plt.savefig(f"{output_folder}residuals_ACF.png")
    plt.clf()

    # Plot the PACF of the residuals
    plt.figure(figsize=(10, 6))
    sm.graphics.tsa.plot_pacf(residuals, lags=lag)
    plt.savefig(f"{output_folder}residuals_PACF.png")
    plt.clf()

    if verbose:
        print("done.")
        print(f"\tResiduals Expected Value = {np.average(residuals)}")



def plot_forecast(time_series:pd.Series, forecast:pd.Series, output_folder:str, verbose:bool=False, model_name:str="ARIMA") -> None:
    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label='Original', color="black", linewidth=2)
    if len(time_series) != len(forecast):
        plt.plot(np.arange(len(time_series), len(time_series)+len(forecast)), forecast, label='Forecast', color="red", linestyle='--')
    else:
        plt.plot(forecast, label='Forecast', color="red", linestyle='--')
    # plt.fill_between(np.arange(len(time_series), len(time_series) + forecast_steps), conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.title(f'{model_name} Model Forecast')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_folder}/{model_name}-forecast.png")
    plt.clf()



def arima_model(p:int,
                i:int,
                q:int,
                train_series:pd.Series|np.ndarray,
                verbose:bool=False,
                color:str="blue",
               ):
    '''
    Initializes and **fits** an ARIMA model with the given parameters, on the given data `train_series`
    '''
    if verbose:
        print("Initializing ARIMA model with the following parameters:")
        utils.print_colored("\tp", color=color, end=f": {p}\n")
        utils.print_colored("\ti", color=color, end=f": {i}\n")
        utils.print_colored("\tq", color=color, end=f": {q}\n")
        print(f"\tTotal: {p+q}")
        print(f"Provided Timeseries has ", end="")
        utils.print_colored(len(train_series), color=color, end=" ")
        print("realizations.")
        print("Fitting started ... ")
    model = tsa.ARIMA(endog=train_series, order=(p, i, q)).fit()
    if verbose:
        print(model.summary())
    return model 



def var_model(p:int, train_series:pd.Series, verbose:bool=False) -> VAR:
    '''
    `train_series` shape must be `(n_samples, n_features)`
    '''
    model = VAR(train_series).fit(p)
    if verbose:
        print(model.summary())
    return model



def check_stationarity(univariate_series:pd.Series|np.ndarray, significance_level:float=0.05, detailed_info:bool=False, verbose:bool=False, print_full_analysis:bool=False) -> bool|int:
    '''
    ADF and KPSS tests for stationarity.
    
    Returns `True` if the series is stationary, `False` otherwise.  The `univariate_series` timeseries must be univariate.

    If `detailed_info` is true it returns the type of (non)stationarity as well:
    - `0`: The series is **stationary**.
    - `1`: The series is **UNIT ROOT** non-stationary.
    - `2`: The series is stationary around a **DETERMINISTIC TREND**
    - `3`: The series is **DIFFERENCE** stationary.
    '''
    adf = adf_test(timeseries=univariate_series, verbose=print_full_analysis)
    adf_stationarity:bool = False

    # ADF checks
    if (adf['p-value'] > adf['Critical Value (5%)']) and (adf['Test Statistic'] > adf['Critical Value (1%)']) and (adf['Test Statistic'] > adf['Critical Value (5%)']) and (adf['Test Statistic'] > adf['Critical Value (10%)']):
        # NON STATIONARY
        adf_stationarity = False
    elif (adf['p-value'] < significance_level):
        # STATIONARY
        adf_stationarity = True
    else:
        adf_stationarity = False
    
    if print_full_analysis:
        print('\n')
    kpss = kpss_test(timeseries=univariate_series, verbose=print_full_analysis)
    kpss_stationarity:bool = False

    # KPSS checks
    if (kpss['p-value'] < significance_level):
        # NON STATIONARY
        kpss_stationarity = False
    else:
        # STATIONARY
        kpss_stationarity = True

    # COMPARE
    if kpss_stationarity and adf_stationarity:
        if verbose:
            print("The series is stationary.")
        return True if not detailed_info else 0
    elif (not kpss_stationarity) and (not adf_stationarity):
        if verbose:
            utils.print_colored("The series is UNIT ROOT, thus not stationary.", 'red')
        return False if not detailed_info else 1
    elif (kpss_stationarity) and (not adf_stationarity):
        if verbose:
            utils.print_colored("The series is stationary around a DETERMINISTIC TREND, use detrending techniques to make it stationary.", 'red')
        return False if not detailed_info else 2
    elif (not kpss_stationarity) and (adf_stationarity):
        if verbose:
            utils.print_colored("The series is DIFFERENCE stationary, use differencing techniques to make it stationary.", 'red')
        return False if not detailed_info else 3
    else:
        raise SyntaxError
    


def diff_timeseries(timeseries:np.ndarray) -> np.ndarray:
    return np.diff(timeseries, axis=0)



def ith_diff_timeseries(timeseries:np.ndarray, i:int=1) -> pd.Series:
    assert i >= 0
    new_series:np.ndarray = timeseries.copy()
    for _ in range(i):
        new_series = diff_timeseries(timeseries=new_series)
    return new_series



def non_stationary_features_list(multivariate_timeseries:pd.Series|np.ndarray,
                                 features:list[str],
                                 detailed_info:bool=False,
                                 verbose:bool=False
                                ) -> list[str]|list[tuple[str,int]]:
    '''
    If `detailed_info` returns a list of tuples `(feature_name, nonstationarity_type)`
    Else returns a list of **non** stationary features
    '''
    non_stationary_features:list[str] = list()
    if verbose:
            print("Stationarity check:")
    for i in range(len(features)):
        feature = features[i]
        if verbose:
            print(f"\t{feature}: ", end="")
        is_stationary = check_stationarity(univariate_series=multivariate_timeseries[:,i], detailed_info=detailed_info, verbose=verbose)
        if detailed_info:
            if not is_stationary > 0:
                non_stationary_features.append((feature,is_stationary))
        else:
            if not is_stationary:
                non_stationary_features.append(feature)
    return non_stationary_features



def adf_test(timeseries:pd.Series|np.ndarray, verbose:bool=False) -> pd.Series:
    '''
    Check stationarity metrics.
    '''
    if verbose:
        print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries)#, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    if verbose:
        print(dfoutput)
    return dfoutput



def kpss_test(timeseries:pd.Series|np.ndarray, verbose:bool=False) -> pd.Series:
    '''
    CHeck for stationarities around deterministic trends
    '''
    import warnings
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    # Ignore only the InterpolationWarning (and keep other warnings)
    warnings.filterwarnings("ignore", category=InterpolationWarning)
    if verbose:
        print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c")#, nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    if verbose:
        print(kpss_output)
    return kpss_output



def check_cointegration(series_1:pd.Series|np.ndarray, series_2:pd.Series|np.ndarray, verbose:bool=False) -> bool:
    '''
    Check if the pair of timeseries can be cointegrated.
    '''
    old_test = OLS(series_1, series_2).fit()
    result = adfuller(old_test.resid)
    t_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    if (t_stat <= critical_values['10%']) and (p_value <= 0.1):
        if verbose:
            print("Pair is co-integrated")
        return True
    else:
        if verbose:
            print("Pair is NOT co-integrated")
        return False
    


def check_multivariate_cointegration(multivariate_timeseries:pd.Series|np.ndarray,
                                     features:list[str],
                                     verbose:bool=False,
                                     color:str="blue",
                                    ) -> list[tuple[str,str]]:
    '''
    Check every pair of features and returns a list of cointegrated features
    '''
    if verbose:
        print("Checking cointegration among", end=" ")
        utils.print_colored(len(features), color=color, end=" ")
        print("timeseries:")
        bar = utils.BAR(len(features)*(len(features)-1))
    cointegrated:list[tuple[str,str]] = list()
    for i in range(len(features)-1):
        for j in range(i+1, len(features)):
            if check_cointegration(series_1=multivariate_timeseries[:,i],
                                   series_2=multivariate_timeseries[:,j],
                                   verbose=False,
                                   ):
                cointegrated.append( (features[i],features[j]) )
                if verbose:
                    bar.update()
    if verbose:
        bar.finish()
        if len(cointegrated) > 0:
            print("The following parameters are cointegrated")
            utils.print_two_column(cointegrated, color=color)
        else:
            print("No timeseries can be cointegrated.")
    return cointegrated



def make_diff_stationary(timeseries:pd.Series|np.ndarray,
                         max_diff:int|None=None,
                         verbose:bool=True
                        ) -> tuple[pd.Series|np.ndarray, int]:
    '''
    Takes the derivative of the timeseries for as long as needed to make it stationary.  It might not always be possible to make a series stationary.

    **Returns**:
    - `stationary_series` if possible else `timeseries`
    - `derivativ_degree` if possible else `-1`
    '''
    from copy import deepcopy
    current_diff:int = 0
    univariate_series = deepcopy(timeseries)
    stationarity:int = 2
    while stationarity in {2, 3}:
        current_diff += 1
        univariate_series = diff_timeseries(univariate_series)
        if verbose:
            print(f"Checking derivative of degree {current_diff}:", end=" ")
        stationarity:int = check_stationarity(univariate_series=univariate_series,
                                              detailed_info=True,
                                              verbose=verbose,
                                              print_full_analysis=False,
                                             )
        if max_diff is not None and current_diff >= max_diff:
            break
    if stationarity == 0:
        # the series is now stationary
        if verbose:
            print(f"The timeseries has been made stationary after {current_diff} differentiations")
        return (univariate_series, current_diff)
    else:
        # the series is not stationary
        if verbose:
            print("Timeseries could not be made stationary with differentiations.")
        return (timeseries, -1)
        


def make_multivariate_diff_stationary(multivariate_timeseries:pd.Series|np.ndarray,
                                      features:list[str],
                                      verbose:bool=False,
                                      max_diff:int|None=None,
                                     ) -> list[tuple[str,str]]:
    from copy import deepcopy
    current_diff:int = 0
    series = deepcopy(multivariate_timeseries)
    stationarity:int = 2
    while stationarity in {2, 3}: # deterministic trend or diff stationarity
        current_diff += 1
        for i in range(len(features)):
            series[1:,i] = diff_timeseries(series[:,i])
        series = series[1:]
        if verbose:
            print(f"Checking derivative of degree {current_diff}:", end=" ")
        non_stat:list[tuple] = non_stationary_features_list(multivariate_timeseries=series,
                                                            features=features,
                                                            verbose=verbose
                                                           )
        
        if len(non_stat) == 0:
            stationarity = 0
        else:
            stationarity = max([s[1] for s in non_stat])

        if max_diff is not None and current_diff >= max_diff:
            break
    if stationarity == 0:
        # the series is now stationary
        if verbose:
            print(f"The multivariate timeseries has been made stationary after {current_diff} differentiations")
        return (series, current_diff)
    else:
        # the series is not stationary
        if verbose:
            print("Multivariate timeseries could not be made stationary with differentiations.")
        return (multivariate_timeseries, -1)