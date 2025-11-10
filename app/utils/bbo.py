import pandas as pd
import nevergrad as ng
import numpy as np
import progressbar
from concurrent import futures
import matplotlib.pyplot as plt
import numpy as np

from typing import Callable
import time
import json



def get_optimizer(instrumentation,
                  optimizer_name:str,
                  num_iterations:int,
                  num_workers:int=1,
                  use_NonObjectOptimizer:bool=False
                 ):
    if not use_NonObjectOptimizer:
        optimizer = ng.optimizers.registry[optimizer_name](instrumentation,
                                                           budget=num_iterations,
                                                           num_workers=num_workers
                                                          )
    else:
        optimizer = ng.families.NonObjectOptimizer(
                                                method=optimizer_name,
                                                random_restart=True
                                                )(instrumentation,
                                                  budget=num_iterations,
                                                  num_workers=num_workers
                                                 )
    return optimizer



def run_bbo(iterations_per_worker:int,
            optimizer_name:str,
            constraints_function:Callable,
            instrumentation:ng.p.Instrumentation,
            obj_function:Callable|None=None,
            use_NonObjectOptimizer:bool=False,
            num_workers:int=1,
            verbose:bool=False,
            target_fitness:float=None,
            MAX_BUDGET:int=None,
            tolerance:int=100,
            bbo_time_limit:int=None,
            plot_loss:bool=True,
            save_plot:bool=False,
            plot_img_folder:str="./"
           ) -> dict:
    """
    Performs the BBO.
    """
    if num_workers > 1:
        if verbose:
            print("Multiple workers not supported for Nevergrad BBO process, reverting to 1 worker.")
        iterations_per_worker = iterations_per_worker * num_workers
        num_workers = 1

    # Get model parameters
    num_iterations = iterations_per_worker*num_workers
    LOSS_HISTORY:list[float] = list()

    if verbose:
        print("BBO has started")
    start = time.time()
    
    optimizer = get_optimizer(instrumentation=instrumentation,
                              optimizer_name=optimizer_name,
                              use_NonObjectOptimizer=use_NonObjectOptimizer,
                              num_iterations=num_iterations,
                              num_workers=num_workers
                             )
        
    if target_fitness is None:
        target_fitness = 0.0
    
    fitness = target_fitness + 1
    best_fitness = 999999999
    tolerated_value = best_fitness
    recommendation = None
    iteration_idx = 0
    current_tolerance = 0
    n_hard_restarts = 0
    while fitness > target_fitness:
        iteration_idx += 1
        x = optimizer.ask()
        fitness = obj_function(*x.args, **x.kwargs)
        if constraints_function is not None:
            constraint_violations = constraints_function(*x.args, **x.kwargs)
            optimizer.tell(x, fitness, constraint_violations)
        else:
            optimizer.tell(x, fitness)
        if verbose:
            print(f"Performing step {iteration_idx}...")

        if fitness < best_fitness:
            best_fitness = fitness
            recommendation = x
            current_tolerance = 0
            # tolerated_value = fitness
        else:
            if fitness >= tolerated_value:
                current_tolerance += 1
                if current_tolerance > tolerance:
                    # random restart
                    tolerated_value = fitness  
                    current_tolerance = 0
                    n_hard_restarts += 1
                    # get new optimizer
                    del optimizer
                    optimizer = get_optimizer(
                                              instrumentation=instrumentation,
                                              optimizer_name=optimizer_name,
                                              use_NonObjectOptimizer=use_NonObjectOptimizer,
                                              num_iterations=num_iterations-iteration_idx,
                                              num_workers=num_workers
                                             )
            else:
                current_tolerance -= 1
                tolerated_value = fitness    

        if plot_loss or save_plot:
            LOSS_HISTORY.append(best_fitness)

        if verbose:
            print(f"Step {iteration_idx} done. (fitness={fitness}" \
                                                + f"{', [NEW!!] ' if best_fitness == fitness else ', '}best={best_fitness}" \
                                                + f', patience={tolerance - current_tolerance}' \
                                                + (f', hard_restarts={n_hard_restarts}' if n_hard_restarts > 0 else '') \
                                                + f', elapsed_time=' + (f'{int((time.time()-start)/60)}min' if int((time.time()-start)/60) < 60 else f'{int((time.time()-start)/60/60)}h{int((time.time()-start)/60)%60}min')
                                                + f', constraints={[round(float(i), 2) for i in constraint_violations]}' if constraints_function is not None else ''\
                                                + ")\n"
                    )
        
        # Conditions for termination
        if MAX_BUDGET is not None:
            # STOP!!
            if iteration_idx >= MAX_BUDGET:
                if verbose:
                    print(f"Max budget ({MAX_BUDGET}) reached with best fitness = {best_fitness}{f' and {n_hard_restarts} random restarts.' if n_hard_restarts > 0 else '.'}")
                break
        if bbo_time_limit is not None:
            # STOP!!
            if time.time()-start > bbo_time_limit:
                if verbose:
                    print(f"Max time ({bbo_time_limit}) reached with best fitness = {best_fitness}{f' and {n_hard_restarts} random restarts.' if n_hard_restarts > 0 else '.'}")
                break
    if verbose:
        print(f"BBO took {int(time.time()-start)}s{'' if int(time.time()-start) < 60 else f' ({int((time.time()-start)/60)} min)'}.")

    if plot_loss or save_plot:
        plt.plot(LOSS_HISTORY)
        plt.grid(True)
        plt.ylabel('Best Loss')
        plt.xlabel('Iteration')
        if plot_loss:
            plt.show()
        if save_plot:
            plt.savefig(f"{plot_img_folder}bbo_loss.png", dpi=300)
        plt.close()
        plt.clf()
        
    return recommendation.kwargs



