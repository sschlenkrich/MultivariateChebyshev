"""
Run Black formula example using MoCaX library for Chebyshev interpolation.

For MoCaX library see
https://www.mocaxintelligence.com/download-mocax-intelligence/
"""

import numpy as np
import pandas as pd
from pprint import pprint
from scipy.stats import norm
import sys
import time

sys.path.append('./')

from src.multivariate_chebyshev_numpy import chebyshev_multi_points  as chebyshev_multi_points_np
from src.multivariate_chebyshev_numpy import chebyshev_transform     as chebyshev_transform_np
from src.multivariate_chebyshev_numpy import chebyshev_coefficients  as chebyshev_coefficients_np
from src.multivariate_chebyshev_numpy import chebyshev_interpolation as chebyshev_interpolation_np

import mocaxpy as mx


def BlackOverK(x):
    moneyness = x[0]
    stdDev    = x[1]
    callOrPut = x[2]
    d1 = np.log(moneyness) / stdDev + stdDev / 2.0
    d2 = d1 - stdDev
    return callOrPut * (moneyness*norm.cdf(callOrPut*d1)-norm.cdf(callOrPut*d2))


def run_performance_testing(perf_degrees, black_function):
    """
    Run the performance testing and return a Dataframe that contains the run times.

    perf_degrees ... a list of Chebyshev polynomial degrees.

    label_dict ... a dictionary of parameters and boundaries

    heston_model_pricer ... a HestonModelPricer object
    """
    a = np.array([ 0.5, 0.01, -1.0 ])
    b = np.array([ 2.0, 0.50, +1.0 ])
    print('Use domain (a,b) = ')
    pprint(a)
    pprint(b)
    np.random.seed(42)
    results = []
    for Nd in perf_degrees:
        degrees = [Nd] * 3  # Chebyshev degrees
        n_points = np.prod([ Nd+1 for Nd in degrees ])
        print('Run Nd = %d with %d Chebyshev points...' % (Nd,n_points))
        res = { 'Nd' : Nd, 'N_Points' : n_points }
        #
        print('  Run Numpy calculations.')
        Z_np = chebyshev_multi_points_np(degrees)
        X_np = chebyshev_transform_np(Z_np, a, b)
        #
        start = time.time()
        values_np  = np.apply_along_axis(black_function, 1, X_np, None)
        end = time.time()
        res['Function_np'] = end - start
        #
        start = time.time()
        C_np = chebyshev_coefficients_np(degrees, Z_np, values_np)
        end = time.time()
        res['Numpy construct'] = end - start
        #
        print('  Run MoCaX calculations.')
        start = time.time()
        model = mx.Mocax(
            black_function,
            len(degrees),
            mx.MocaxDomain([[a_,b_] for a_, b_ in zip(a,b)]),
            None,
            mx.MocaxNs(degrees),
            max_derivative_order=0,
        )
        end = time.time()
        res['MoCaX construct'] = end - start
        derivativeId = model.get_derivative_id([0]*3)
        start = time.time()
        y_mocax = np.apply_along_axis(model.eval, 1, X_np, derivativeId)
        #y_mocax = np.array([
        #    model.eval(x.tolist(), derivativeId) for x in X_np
        #])
        end = time.time()
        res['MoCaX Cheb. eval'] = end - start
        res['MoCaX check'] = np.max(np.abs(y_mocax - values_np))
        # We also test random points...
        X_rn = a + np.random.uniform(size=(len(X_np), 3)) * (b-a)
        #
        start = time.time()
        V_rn = np.apply_along_axis(black_function, 1, X_rn, None)
        res['Function_rn'] = time.time() - start
        #
        start = time.time()
        V_np = chebyshev_interpolation_np(X_rn, C_np, a, b)
        res['Numpy_rn'] = time.time() - start
        #
        start = time.time()
        V_mx = np.apply_along_axis(model.eval, 1, X_rn, derivativeId)
        #V_mx = np.array([
        #    model.eval(x.tolist(), derivativeId) for x in X_rn
        #])
        res['MoCaX_rn'] = time.time() - start
        #
        res['Numpy approx'] = np.max(np.abs(V_np - V_rn))
        res['MoCaX approx'] = np.max(np.abs(V_mx - V_rn))
        res['MoCaX check rn.'] = np.max(np.abs(y_mocax - values_np))
        #
        results.append(res)
    results = pd.DataFrame(results)
    print('Done.')
    return results


if __name__ == '__main__':
    black_function = lambda x, p : BlackOverK(x)
    root_path = './'  # assume we start script from project root
    # we add some warm-up calculations at the beginning
    res = run_performance_testing([
         4,  4,  4,
         8,  8,  8,
        12, 12, 12,
        16, 16, 16,
        20, 20, 20,
        ],
        black_function)
    res.to_csv(root_path + 'examples/black/black_mocax_performance.csv', sep=';')
    print(res)

