import lhsmdu # hypercube generator
import numpy as np

from typing import Any, Callable

import utils.utils as utils


def hypercube_optimization(iterations:int,
                           parameters_bounds:dict[str,tuple[float|int]|list[float|int]],
                           model_score_f:Callable,
                           static_params:dict|None=None,
                           save_partial_results:str|None=None,
                           model_name:str="model",
                           n_workers:int=1,
                           verbose:bool=True,
                          ) -> dict[str,Any]:
    """
    Performs *hypercube optimization* on the parameters specified in `parameter_bounds`, each parameter space will only be searched within the **provided bounds**.
    
    **Arguments**:
    - `iterations` : number of samples from the parameter space to test for the best setting.
    - `parameters_bounds` : dictionary containing each parameter to optimize, the format must be
        - `"parameter_name" : (lower_bound,upper_bound)`
    - `static_params` : extra **optional** parameters not to be optimized that are constant throughout the optimization process
    - `model_score_f` : function to test the suggested parameters and return a `float` with the **score** *(to maximize)*
    - `save_partial_results` : if `None` no *intermediate result* is saved, otherwise partial *best* results will be saved at the provided path (as .json files) as the optimization is still going
    - `model_name` : identifier for saving files
    - `n_workers` : amount of parallel cores
    - `verbose` : whether to print status messages in **std** 

    **Returns**:
    - dictionary of hyperparameters.
    """
    if verbose:
        utils.print_colored("Latin Hypercube Optimization", highlight="yellow")
        print(f"Generating hypercube for", end=" ")
        utils.print_colored(iterations, color="blue", end=" ")
        print("samples ... ", end="")
    latin_space = np.array(lhsmdu.sample(len(parameters_bounds.keys()),iterations))
    if verbose:
        print("done.")
    # generate ids for parameters
    parameter_id = dict()
    id = 0
    for param in sorted(list(parameters_bounds.keys())):
        parameter_id[param] = id
        id += 1
    BEST_SCORE:float = -999999.0
    BEST_HYPERPARAMETERS:dict = dict()
    iter:int = 0    
    if verbose:
        print(f"Optimizing the following hyperparameters:")
        utils.print_dict(parameters_bounds, key_color="blue", value_color="grey")
        bar = utils.BAR(iterations)
    while iter < iterations:
        concurrent_params:list[dict[str,float]] = list()
        for _ in range(n_workers):
            try:
                # get parameter values
                if static_params is not None:
                    current_params = static_params.copy()
                else:
                    current_params:dict[str] = dict()
                for p,v in parameters_bounds.items():
                    current_params[p] = latin_space[parameter_id[p],iter]*(v[1]-v[0]) + v[0]
                iter += 1
            except IndexError:
                break
            concurrent_params.append(current_params)
        # get trained model        
        concurrent_results = utils.embarassing_parallelism(function=model_score_f,
                                                           n_workers=len(concurrent_params),
                                                           arguments_list=concurrent_params,
                                                           use_process=False
                                                          )
        # check against best score
        for result in concurrent_results:
            current_score = result['score']
            if current_score > BEST_SCORE:
                # update
                BEST_SCORE = current_score
                for k in parameters_bounds.keys():
                    BEST_HYPERPARAMETERS[k] = result['params'][k]
                if save_partial_results is not None:
                    utils.save_json(BEST_HYPERPARAMETERS, f"{save_partial_results}{model_name}-PARTIAL.json")
        if verbose:
            bar.update(status=iter)
    if verbose:
        bar.finish()
    return BEST_HYPERPARAMETERS





   