import lhsmdu
import numpy as np
from typing import Any

import utils.utils as utils



def get_model_score(params:dict[str,float]) -> dict[str,Any]:
    '''
    Evaluates the model using the given params and returns the score and the input parameters
    
    The **SCORE** is meant to be **maximized**!

    **Arguments**:
    - `params` : dictionary with the following format
        - `"param_name" : suggested_value`
    
    **Returns**:
    - A dictionary with the following data
        - `score` : `float`
        - `params` : `dict`
    '''
    import marl
    # needed for the `marl.get_agent_env` and `marl.single_model_train` functions
    from uav_inertia.pettingzoo_env.evader_env import ModelicaEvaders
    from uav_inertia.pettingzoo_env.pursuer_env import ModelicaPursuers 
    from uav_inertia.pettingzoo_env.multi_agent_env import ModelicaEvaderPursuers
    model_folder = params['static_params']['inertia_model']
    n_layers:int = params['n_layers']
    hidden_size:int = params['hidden_size']
    net_arch:list[dict[str,list[int]]] = [{
                                           'pi': [hidden_size for _ in range(n_layers)], # actor (policy)
                                           'vf': [hidden_size for _ in range(n_layers)], # critic (value function)
                                         }]
    P:dict = params['static_params']
    C:dict = P['case_study_params'][P['case_study']]

    agent, env = marl.get_agent_env(get_evader=True,
                                    modelica_model_folder=model_folder,
                                    modelica_time_steps=P['action_frequency'],
                                    hit_radius=C['hit_radius'],
                                    grid_x=C['grid_x'],
                                    grid_y=C['grid_y'],
                                    grid_z=C['grid_z'],
                                    max_steps=C['episode_len'],
                                    model_path=None,
                                    adversary_model_path=None,
                                    caught_constant=P['caught_constant'],
                                    crash_constant=P['crash_constant'],
                                    time_constant=P['time_constant'],
                                    net_arch=net_arch,
                                    img_folder=P['img_folder'],
                                    case_study=P['case_study'],
                                    verbose=False,
                                   )
    _, _, score, _ = marl.single_model_train(agent=agent,
                                             env=env,
                                             model_save_folder=None,
                                             episodes=P['training_epochs'],
                                             test_episodes=P['n_tests'],
                                             verbose=False,
                                             render=False,
                                             SMC=False,
                                            )
    return {'score': score,
            'params' : params
           }



def hypercube_optimization(iterations:int,
                           parameters_bounds:dict[str,tuple[float|int]|list[float|int]],
                           static_params:dict|None=None,
                           model_score_f=get_model_score,
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
        utils.print_colored(hypercupe_iterations, color="blue", end=" ")
        print("samples ... ", end="")
    latin_space = np.array(lhsmdu.sample(len(parameters_bounds.keys()),hypercupe_iterations))
    if verbose:
        print("done.")
    # generate ids for parameters
    parameter_id = dict()
    id = 0
    for param in sorted(list(parameters_bounds.keys())):
        parameter_id[param] = id
        id += 1
    BEST_SCORE:float = -999999.0
    BEST_HYPERPARAMETERS:dict = parameters_bounds
    iter:int = 0    
    if verbose:
        print(f"Optimizing the following hyperparameters:")
        utils.print_dict(parameters_bounds, key_color="blue", value_color="grey")
        bar = utils.BAR(hypercupe_iterations)
    while iter < iterations:
        concurrent_params:list[dict[str,float]] = list()
        for _ in range(n_workers):
            try:
                # get parameter values
                current_params:dict[str] = dict()
                for p,v in parameters_bounds.items():
                    current_params[p] = latin_space[parameter_id[p],iter]*(v[1]-v[0]) + v[0]
                if static_params is not None:
                    current_params.update(static_params)
                # update variables
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
                BEST_HYPERPARAMETERS = result['params']
                if save_partial_results is not None:
                    utils.save_json(BEST_HYPERPARAMETERS, f"{save_partial_results}{model_name}-PARTIAL.json")
        if verbose:
            bar.update(k=iter)
    if verbose:
        bar.finish()
    return BEST_HYPERPARAMETERS




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    # = PARAMETERS = #
    parameters_file:str = "/data/marl_params.json"
    save_partial_results:bool = True


    parameters:dict = utils.load_json(parameters_file)
    verbose:bool = parameters['verbose']
    utils.print_colored(f"Hyper-Parameter Optimization ({parameters['case_study']})", highlight="magenta")    

    # # # # # # # # #
    # OPTIMIZATION! #
    # # # # # # # # #

    BEST_HYPERPARAMETERS = hypercube_optimization(iterations=parameters['optimization_steps'],
                                                  parameters_bounds=parameters['parameters_bounds'],
                                                  static_params=parameters,
                                                  model_score_f=get_model_score,
                                                  save_partial_results=parameters['optimization_folder'],
                                                  model_name=parameters['case_study'],
                                                  n_workers=parameters['n_cores'],
                                                  verbose=verbose,
                                                 )
    utils.save_json(BEST_HYPERPARAMETERS, f"{parameters['optimization_folder']}{parameters['case_study']}-hyperparameters.json")

    # SHOW TESTS

    if verbose:
        print(f"Best parameters found:")
        utils.print_dict(BEST_HYPERPARAMETERS)

   