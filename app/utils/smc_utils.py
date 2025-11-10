import math
from copy import deepcopy
from typing import Literal, Callable, Any

from scipy.special import betaincinv

import utils.utils as utils


def okamoto(eps, delta):
    # the Okamoto bound
    return math.ceil(math.log(2 / delta) / (2 * eps * eps))


def cp_int(N, Np, delta, side:Literal['left', 'right', 'both']='both') -> tuple|float:
    """
    Clopper_pearson confidence interval.
    """
    assert side in ['left', 'right', 'both']
    if side != 'right':
        if Np == 0:
            cp_l = 0
        else:
            cp_l = betaincinv(Np, N - Np + 1, delta/2)
        if side == 'left':
            return cp_l
    if Np == N:
        cp_h = 1
    else:
        cp_h = betaincinv(Np + 1, N - Np, 1 - delta / 2)
    if side == 'right':
        return cp_h
    else:
        return cp_l, cp_h


def intersect(a, b, c, d):
    """return the length of intersection of `[a,b]` and `[c,d]`"""
    return max(min(b, d) - max(a, c), 0)


def margin_left(n:int, delta:float, a:float) -> int:
    """Finds `k`, such that `CP(n,k,'left') <= a`, `CP(n,k+1,'left') > a`"""
    assert cp_int(n, n, delta, 'left') > a
    low = 0
    high = n
    while high - low > 1:
        mid = (high + low) // 2
        if cp_int(n, mid, delta, 'left') > a:
            high = mid
        else:
            low = mid
    return low


def margin_right(n:int, delta:float, b:float) -> int:
    """Finds `k`, such that `CP(n,k,'right') <= b`, `CP(n,k+1,'right') > b`"""
    assert cp_int(n, 0, delta, 'right') < b
    low = 0
    high = n
    while high - low > 1:
        mid = (high + low) // 2
        if cp_int(n, mid, delta, 'right') >= b:
            high = mid
        else:
            low = mid
    return low


def max_length(n:int, delta:float, a:float, b:float):
    """Algorithm 2 in paper.
    
    Returns the max length of CP(n,k,delta)∩[a,b], k=0,1,...,n
    """
    if cp_int(n, 0, delta, 'right') >= b or cp_int(n, n, delta, 'left') <= a:
        return b - a
    k1 = margin_left(n, delta, a)
    k2 = margin_right(n, delta, b)
    l1, h1 = cp_int(n, k1, delta)
    l2, h2 = cp_int(n, k1 + 1, delta)
    l3, h3 = cp_int(n, k2, delta)
    l4, h4 = cp_int(n, k2 + 1, delta)
    l5, h5 = cp_int(n, n // 2, delta)
    return max(intersect(a, b, l1, h1), intersect(a, b, l2, h2), intersect(a, b, l3, h3), intersect(a, b, l4, h4),
               intersect(a, b, l5, h5))


def interval_sensitive_bound(eps:float, delta:float, a:float, b:float):
    """Algorithm 1 in paper"""
    if b - a <= 2 * eps:
        return 0
    low = 0
    high = okamoto(eps, delta)
    while high - low > 1:
        mid = (high + low) // 2
        if max_length(mid, delta, a, b) <= 2 * eps:
            high = mid
        else:
            low = mid
    return high


def validate(n:int, delta:float, a:float, b:float, eps:float, verbose:bool=False) -> bool:
    """Algorithm 3 in paper

    Validate if the max length of CP(n,k)∩[a,b] <= 2*eps (k=0,1,...,n)
    """
    if cp_int(n, 0, delta, 'right') >= b or cp_int(n, n, delta, 'left') <= a:
        if b - a <= 2 * eps:
            return True
        else:
            if verbose:
                print(f'WARNING! Validate failure 1: n={n};, delta={delta}; [a,b]=[{a},{b}]; epsilon={eps}')
            return False
    k1 = margin_left(n, delta, a)
    k2 = margin_right(n, delta, b)
    if k1 > k2 + 1:
        if b - a <= 2 * eps:
            return True
        else:
            if verbose:
                print(f'WARNING! Validate failure 2: n={n};, delta={delta}; [a,b]=[{a},{b}]; epsilon={eps}')
            return False
    for i in range(k1, k2 + 1, 1):
        l, h = cp_int(n, i, delta)
        if intersect(a, b, l, h) > 2 * eps:
            if verbose:
                print(f'WARNING! Validate failure 3: n={n};, delta={delta}; [a,b]=[{a},{b}]; epsilon={eps}')
            return False
    return True


def estimate(n:int,
             delta:float,
             a:float,
             b:float,
             eps:float,
             property:Callable,
             model:Any,
             verbose:bool=True,
            ) -> tuple[float, list[dict]|None]:
    """Algorithm 4 in paper
    
    Returns the estimation of probability of the model `model` satisfying the property in `property` and the counterexampes found (if any).

    **Returns:** (`probability`, `counterexamples`)
    """
    all_counterexamples:list[dict] = list()
    if n == 0:
        return (a + b) / 2, None
    
    if not validate(n, delta, a, b, eps):
        est_count = 0
        n1 = okamoto(eps, delta)
        if verbose:
            print("Evaluating with ", end="")
            utils.print_colored("Okamoto", color="light_red", end=":\n")
            bar = utils.BAR(n1)
        for _ in range(n1):
            # run n (okamoto) simulations and check the property
            satisfied, counterexample = property(model)
            if satisfied:
                est_count += 1
            else:
                all_counterexamples.append(counterexample)
            if verbose:
                bar.update()
        return est_count / n1, all_counterexamples if len(all_counterexamples) > 0 else None
    
    else:
        est_count = 0
        if verbose:
            print("Evaluating with ", end="")
            utils.print_colored("Clopper-Pearson", color="light_green", end=":\n")
            bar = utils.BAR(n)
        for _ in range(n):
            # run n (clopper-pearson) simulations and check the property
            satisfied, counterexample = property(model)
            if satisfied:
                est_count += 1
            else:
                all_counterexamples.append(counterexample)
            if verbose:
                bar.update()
        lb, ub = cp_int(n, est_count, delta)

        if ub <= a:
            return a, all_counterexamples if len(all_counterexamples) > 0 else None
        elif lb >= b:
            return b, all_counterexamples if len(all_counterexamples) > 0 else None
        else:
            return (max(lb, a) + min(ub, b)) / 2, all_counterexamples if len(all_counterexamples) > 0 else None



# NOTE: this is the complete algorithm from the paper and the only function that should be used
def adaptive_estimation(eps:float,
                        delta:float,
                        model:Callable,
                        property:Callable=property,
                        a:float=0.0,
                        b:float=1.0,
                        verbose:bool=True,
                       ) -> tuple[float, list[dict]|None]:
    """
    **Input:** precision `eps` and confidence `delta`

    **Output:** estimated probability of satisfying the property
    """
    all_counterexamples:list[dict] = list()
    delta1 = delta * 0.05
    n1 = interval_sensitive_bound(eps, delta, a, b)
    n2 = max(min(math.ceil(0.01 * n1), 100), 10)
    simulation_num = n2
    count_1 = 0
    if verbose:
        print("Starting ", end="")
        utils.print_colored("initial", color="light_blue", end=" ")
        print(f"estimation (epsilon={eps}; delta={delta}):")
        bar = utils.BAR(n2)
    for _ in range(n2):
        satisfied, counterexample = property(model)
        if satisfied:
            count_1 += 1
        else:
            all_counterexamples.append(counterexample)
        if verbose:
            bar.update()
    p_1 = count_1 / n2

    n_list = [round((i + 1) * n1 / 100) for i in range(20)]
    length = len(n_list)
    cost = n_list.copy()
    for i in range(length):
        n3 = n_list[i]
        Np = round(n3 * p_1)
        cp_l, cp_h = cp_int(n3, Np, delta1)
        cost[i] += interval_sensitive_bound(eps, (delta - delta1) / (1 - delta1), cp_l, cp_h)
    m = min(cost)
    if m > n1:
        adaptive_result, Cs = estimate(n=n1, delta=delta, a=a, b=b, eps=eps, property=property, model=model, verbose=verbose)
        if Cs is not None:
            all_counterexamples += Cs
        simulation_num += n1
    else:
        n4 = n_list[cost.index(m)]
        simulation_num += n4
        count_2 = 0
        if verbose:
            utils.print_colored("Additional", color="light_yellow", end=" ")
            print("estimation:")
            bar.reset(n4)
        for _ in range(n4):
            satisfied, counterexample = property(model)
            if satisfied:
                count_2 += 1
            else:
                all_counterexamples.append(counterexample)
            if verbose:
                bar.update()
        cp_l, cp_h = cp_int(n4, count_2, delta1)
        n5 = interval_sensitive_bound(eps, (delta - delta1) / (1 - delta1), cp_l, cp_h)
        simulation_num += n5
        adaptive_result, Cs = estimate(n=n5, delta=(delta - delta1) / (1 - delta1), a=cp_l, b=cp_h, eps=eps, property=property, model=model, verbose=verbose)
        if Cs is not None:
            all_counterexamples += Cs
    if verbose:
        print(f"Total number of simulations: ", end="")
        utils.print_colored(simulation_num, color="blue")
        print(f"Estimated Probability is: ", end="")
        if adaptive_result <= 0.3:
            utils.print_colored(adaptive_result, color="light_red")
        elif adaptive_result <= 0.6:
            utils.print_colored(adaptive_result, color="light_yellow")
        else:
            utils.print_colored(adaptive_result, color="light_green")
    return adaptive_result, all_counterexamples if len(all_counterexamples) > 0 else None



def general_eval(n:int, property:(...), model:(...), n_workers:int=1,verbose:bool=False) -> tuple[int, list[dict]]:
    """
    
    """
    args:list = [{'p':property, 'm':model()} for _ in range(n_workers)]
    f:Callable = lambda a: a['p'](a['m'])
    i:int = 0
    true_count:int = 0
    counterexamples:list[dict] = list()
    if verbose:
        bar = utils.BAR(n)
    while i < n:
        processes = min(n_workers, n-i)
        results = utils.embarassing_parallelism(function=f,
                                                n_workers=processes,
                                                arguments_list=args[:processes],
                                                use_process=False,
                                                )
        i += processes
        for satisfied, counterexample in results:
            if satisfied:
                true_count += 1
            else:
                counterexamples.append(counterexample)
            if verbose:
                bar.update()
    return true_count, counterexamples


def parallel_estimate(n:int,
                      delta:float,
                      a:float,
                      b:float,
                      eps:float,
                      property:Callable,
                      model:Any,
                      n_workers:int=1,
                      verbose:bool=True,
                     ) -> tuple[float, list[dict]|None]:
    """Algorithm 4 in paper
    
    Returns the estimation of probability of the model `model` satisfying the property in `property` and the counterexampes found (if any).

    This function is the (embarassingly) parallelized verion of `estimate`

    **Returns:** (`probability`, `counterexamples`)
    """
    assert n_workers >= 1
    all_counterexamples:list[dict] = list()
    if n == 0:
        return (a + b) / 2, None
    
    if not validate(n, delta, a, b, eps):
        est_count = 0
        n1 = okamoto(eps, delta)
        if verbose:
            print("Evaluating with ", end="")
            utils.print_colored("Okamoto", color="light_red", end=":\n")
        est_count, counterexamples = general_eval(n=n1, property=property, model=model, n_workers=n_workers, verbose=verbose)
        all_counterexamples += counterexamples
        return est_count/n1, all_counterexamples if len(all_counterexamples) > 0 else None
    
    else:
        est_count = 0
        if verbose:
            print("Evaluating with ", end="")
            utils.print_colored("Clopper-Pearson", color="light_green", end=":\n")
        est_count, counterexamples = general_eval(n=n, property=property, model=model, n_workers=n_workers, verbose=verbose)
        all_counterexamples += counterexamples
        # compute clopper-pearson interval
        lb, ub = cp_int(n, est_count, delta)

        if ub <= a:
            return a, all_counterexamples if len(all_counterexamples) > 0 else None
        elif lb >= b:
            return b, all_counterexamples if len(all_counterexamples) > 0 else None
        else:
            return (max(lb, a) + min(ub, b)) / 2, all_counterexamples if len(all_counterexamples) > 0 else None




def parallel_adaptive_estimation(eps:float,
                                 delta:float,
                                 model:Callable,
                                 property:Callable,
                                 a:float=0.0,
                                 b:float=1.0,
                                 n_workers:int=1,
                                 verbose:bool=True,
                                ) -> tuple[float, list[dict]|None]:
    """
    **Input:** precision `eps` and confidence `delta`

    This function is the (embarassingly) parallelized version of `adaptive_estimation` 

    **Output:** estimated probability of satisfying the property
    """
    assert n_workers >= 1
    all_counterexamples:list[dict] = list()
    delta1 = delta * 0.05
    n1 = interval_sensitive_bound(eps, delta, a, b)
    n2 = max(min(math.ceil(0.01 * n1), 100), 10)
    simulation_num = n2
    if verbose:
        print("Starting ", end="")
        utils.print_colored("initial", color="light_blue", end=" ")
        print(f"estimation (epsilon={eps}; delta={delta}):")
    count_1, counterexamples = general_eval(n=n2, property=property, model=model, n_workers=n_workers, verbose=verbose)
    all_counterexamples += counterexamples
    p_1 = count_1 / n2

    n_list = [round((i + 1) * n1 / 100) for i in range(20)]
    length = len(n_list)
    cost = n_list.copy()
    for i in range(length):
        n3 = n_list[i]
        Np = round(n3 * p_1)
        cp_l, cp_h = cp_int(n3, Np, delta1)
        cost[i] += interval_sensitive_bound(eps, (delta - delta1) / (1 - delta1), cp_l, cp_h)
    m = min(cost)
    if m > n1:
        adaptive_result, Cs = parallel_estimate(n=n1, delta=delta, a=a, b=b, eps=eps, property=property, model=model, n_workers=n_workers, verbose=verbose)
        if Cs is not None:
            all_counterexamples += Cs
        simulation_num += n1
    else:
        n4 = n_list[cost.index(m)]
        simulation_num += n4
        count_2 = 0
        if verbose:
            utils.print_colored("Additional", color="light_yellow", end=" ")
            print("estimation:")
        count_2, counterexamples = general_eval(n=n4, property=property, model=model, n_workers=n_workers, verbose=verbose)
        all_counterexamples += counterexamples
        cp_l, cp_h = cp_int(n4, count_2, delta1)
        n5 = interval_sensitive_bound(eps, (delta - delta1) / (1 - delta1), cp_l, cp_h)
        simulation_num += n5
        adaptive_result, Cs = parallel_estimate(n=n5, delta=(delta - delta1) / (1 - delta1), a=cp_l, b=cp_h, eps=eps, property=property, model=model, n_workers=n_workers, verbose=verbose)
        if Cs is not None:
            all_counterexamples += Cs
    if verbose:
        print(f"Total number of simulations: ", end="")
        utils.print_colored(simulation_num, color="blue")
        print(f"Estimated Probability is: ", end="")
        if adaptive_result <= 0.3:
            utils.print_colored(adaptive_result, color="light_red")
        elif adaptive_result <= 0.6:
            utils.print_colored(adaptive_result, color="light_yellow")
        else:
            utils.print_colored(adaptive_result, color="light_green")
    return adaptive_result, all_counterexamples if len(all_counterexamples) > 0 else None
