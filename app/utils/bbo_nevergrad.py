import numpy as np
import nevergrad as ng
import matplotlib.pyplot as plt

from typing import Literal, Callable

import utils.utils as utils



def avg_parameter_magnitude(parameters) -> float:
    avg_param_magnitude:float = 0
    for i in list(parameters.values()):
        avg_param_magnitude += abs(i)
    return avg_param_magnitude / len(parameters)



def obj_function(parameters:dict,
                 saturation:float=500,
                 use_constraints:bool=True,
                 verbose:bool=False,
                ) -> float|tuple[float,list[float]]:
    """
    This function takes the parameters from Nevergrad, formats them and computes 
    the loss function **to minimize**.
    
    **Returns**: `kpi`, `constraints`
    """
    # FORMAT PARAMETERS
    x = parameters['x']
    # SIMULATE
    result = 0.0
    # COMPUTE KPI
    kpi = result
    # CONSTRAINTS
    constraint = constraint_template(result)
    constraint_violations = [constraint]
    # END
    kpi = min(kpi, saturation) if not np.isnan(kpi) else saturation
    if use_constraints:
        return kpi, constraint_violations
    else: 
        return kpi



def constraint_template(model_results:dict) -> float:
    '''
    How to interpret this function

    - \>= 0 if ok

    - < 0 if violated
    '''
    return 0.0


def get_optimizer(instrumentation,
                  optimizer_name:str,
                  num_iterations:int,
                  num_workers:int=1,
                  use_NonObjectOptimizer:bool=False
                 ):
    if not use_NonObjectOptimizer:
        return ng.optimizers.registry[optimizer_name](instrumentation,
                                                      budget=num_iterations,
                                                      num_workers=num_workers
                                                     )
    else:
        return ng.families.NonObjectOptimizer(method=optimizer_name,
                                              random_restart=True
                                              )(instrumentation,
                                                budget=num_iterations,
                                                num_workers=num_workers
                                               )


def bbo_instrumentation_template(verbose:bool=False) -> ng.p.Instrumentation:
    """
    Returns the parameters formatted as Nevergrad requires.
    """ 
    if verbose:
        print("Loading instrumentation...", end="")
    instrumentation = ng.p.Instrumentation(  
            a_days = ng.p.Scalar(init=np.random.uniform(low=0, high=1), lower=0, upper=1),
        )
    if verbose:
        print("done.")
    return instrumentation



def run_bbo(function:Callable,
            iterations_per_worker:int,
            optimizer_name:str,
            instrumentation:ng.p.Instrumentation,
            use_NonObjectOptimizer:bool=False,
            num_workers:int=1,
            verbose:bool=False,
            target_fitness:float|None=None,
            MAX_BUDGET:int|None=None,
            tolerance:int|None=None,
            time_limit:int|None=None,
            plot_folder:str|None=None,
            max_possible_fitness:float=999999.9,
            color:str="blue",
            ) -> dict:
    """
    Performs the BBO.

    **Arguments**:
    - `function` : the function to **minimize**
    - `iterations_per_worker` : how many BBO iterations each worker must do
    - `optimizer_name` : nameof the **Nevergrad** Optimizer to use
    - `use_NonObjectOptimizer` : whether to use *NonObjectOptimizers*
    - `num_workers` : number of parallel workers
    - `target_fitness` : if provided, then th optimization will end **as soon as** the provided fitness value (output of `function`) is reached
    - `MAX_BUDGET` : if provided, no more than `MAX_BUDGET` BBO iterations will be allowed
    - `tolerance` : if provided, when the yielded fitness (output of `function`) doesn't improve for the given amount of steps, then **the optimizer is reset**
    - `time_limit` : if provided, BBO will be forced to stop after the provided amount of **seconds**
    - `plot_folder` : if provided, the fitness (output of `function`) over time will be plotted and saved at the given path
    - `max_possible_fitness` : upper bound of the yielded firness (output of `function`)
    - `verbose` : whether or not to print status messages
    - `color` : which color to use for the prints
    """
    num_iterations:int = iterations_per_worker*num_workers
    if plot_folder is not None:
        LOSS_HISTORY:list[float] = list()
    if verbose:
        print("BBO has started")
    timer = utils.TimeExecution()
    timer.start()
    optimizer = get_optimizer(instrumentation=instrumentation,
                              optimizer_name=optimizer_name,
                              use_NonObjectOptimizer=use_NonObjectOptimizer,
                              num_iterations=num_iterations,
                              num_workers=num_workers
                             )
    if target_fitness is None:
        # perform optimization only up to budget iterations
        # TODO: better parallelize this
        recommendation = optimizer.minimize( objective_function=function, verbosity=int(verbose) )
    else:
        fitness:float = target_fitness + 1
        best_fitness = max_possible_fitness
        tolerated_value = best_fitness
        recommendation = None
        iteration_idx:int = 0
        current_tolerance:int = 0
        n_hard_restarts:int = 0
        while fitness > target_fitness:
            processes = min(num_workers, num_workers if MAX_BUDGET is None else MAX_BUDGET-iteration_idx)
            iteration_idx += processes
            if verbose:
                print(f"Performing optimization step", end=" ")
                utils.print_colored(iteration_idx, color=color, end=" ")
                print((f"({round(iteration_idx/MAX_BUDGET*100,2)}%)" if MAX_BUDGET is not None else ""), end=" ... ")
            # COMPUTE FITNESS
            parallel_parameters = list()
            for _ in range(num_workers):
                x = optimizer.ask()
                parallel_parameters.append(x)
            all_results = utils.embarassing_parallelism(function,
                                                        num_workers,
                                                        [x.kwargs for x in parallel_parameters],
                                                        use_kwargs=True,
                                                        use_process=True,
                                                       )
            # res = function(*x.args, **x.kwargs)
            for i in range(len(all_results)):
                res = all_results[i]
                x = parallel_parameters[i]
                try:
                    fitness, constraint_violations = res
                    optimizer.tell(x, fitness, constraint_violations)
                except:
                    fitness = res
                    constraint_violations = None
                    optimizer.tell(x, fitness)
            fitness = min(all_results)
            x = parallel_parameters[all_results.index(fitness)]
            # PROCESS RESULTS
            if plot_folder is not None:
                LOSS_HISTORY.append(fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                recommendation = x
                current_tolerance = 0
                # tolerated_value = fitness # NOTE: hard tolerance
            else:
                if fitness >= tolerated_value:
                    current_tolerance += 1
                    if tolerance is not None and current_tolerance > tolerance:
                        # random restart
                        tolerated_value = fitness  
                        current_tolerance = 0
                        n_hard_restarts += 1
                        # get new optimizer
                        del optimizer
                        optimizer = get_optimizer(instrumentation=instrumentation,
                                                  optimizer_name=optimizer_name,
                                                  use_NonObjectOptimizer=use_NonObjectOptimizer,
                                                  num_iterations=num_iterations-iteration_idx,
                                                  num_workers=num_workers,
                                                 )
                else:
                    current_tolerance = max(0, current_tolerance-1)
                    tolerated_value = fitness    
            if verbose:
                # TODO: format this better
                print(f"done.\n\tfitness: {fitness}", end="\n\t")
                utils.print_colored("best", color=color, end=": ")
                print(best_fitness, end=" ")
                if best_fitness == fitness:
                    utils.print_colored("[NEW!!]", highlight=color, end="")
                if tolerance is not None:
                    print(f"\n\tpatience: {tolerance - current_tolerance}", end="")
                if n_hard_restarts > 0:
                    print("\n\t", end="")
                    utils.print_colored("hard restarts", color=color, end=": ")
                    print(n_hard_restarts, end="")
                print("\n")
            # TERMINATION CONDITIONS
            if MAX_BUDGET is not None:
                if iteration_idx >= MAX_BUDGET:
                    if verbose:
                        utils.print_colored(f"Max budget ({MAX_BUDGET}) reached with best fitness = {best_fitness}{f' and {n_hard_restarts} random restarts.' if n_hard_restarts > 0 else '.'}", highlight=color)
                    break
            if time_limit is not None:
                if timer.elapsed() > time_limit:
                    if verbose:
                        utils.print_colored(f"Max time ({time_limit}s) reached with best fitness = {best_fitness}{f' and {n_hard_restarts} random restarts.' if n_hard_restarts > 0 else '.'}", highlight=color)
                    break
    if verbose:
        timer.end()
        timer.print()
    if plot_folder is not None:
        plt.plot(sorted(LOSS_HISTORY, reverse=True))
        plt.ylabel('Loss')
        plt.savefig(f"{plot_folder}/bbo_loss.png", dpi=200)
    return recommendation.kwargs






    