"""
Run Heston example using MoCaX library for Chebyshev interpolation.

For MoCaX library see
https://www.mocaxintelligence.com/download-mocax-intelligence/
"""

import numpy as np
import pandas as pd
from pprint import pprint
import sys
import time

sys.path.append('./')

from heston_model_pricer import implied_volatility_from_vector

from src.multivariate_chebyshev_numpy import chebyshev_multi_points  as chebyshev_multi_points_np
from src.multivariate_chebyshev_numpy import chebyshev_transform     as chebyshev_transform_np
from src.multivariate_chebyshev_numpy import chebyshev_coefficients  as chebyshev_coefficients_np
from src.multivariate_chebyshev_numpy import chebyshev_interpolation as chebyshev_interpolation_np

import mocaxpy as mx


def run_performance_testing(perf_degrees, label_dict, heston_function):
    """
    Run the performance testing and return a Dataframe that contains the run times.

    perf_degrees ... a list of Chebyshev polynomial degrees.

    label_dict ... a dictionary of parameters and boundaries

    heston_model_pricer ... a HestonModelPricer object
    """
    print('Use label_dict = ')
    pprint(label_dict)
    a = np.array([label_dict[k][0] for k in label_dict])
    b = np.array([label_dict[k][1] for k in label_dict])
    np.random.seed(42)
    results = []
    for Nd in perf_degrees:
        degrees = [Nd] * 8  # Chebyshev degrees
        n_points = np.prod([ Nd+1 for Nd in degrees ])
        print('Run Nd = %d with %d Chebyshev points...' % (Nd,n_points))
        res = { 'Nd' : Nd, 'N_Points' : n_points }
        #
        print('  Run Numpy calculations.')
        Z_np = chebyshev_multi_points_np(degrees)
        X_np = chebyshev_transform_np(Z_np, a, b)
        #
        start = time.time()
        values_np  = np.apply_along_axis(heston_function, 1, X_np, None)
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
            heston_function,
            len(degrees),
            mx.MocaxDomain([[a_,b_] for a_, b_ in zip(a,b)]),
            None,
            mx.MocaxNs(degrees),
            max_derivative_order=0,
        )
        end = time.time()
        res['MoCaX construct'] = end - start
        derivativeId = model.get_derivative_id([0]*8)
        start = time.time()
        y_mocax = np.apply_along_axis(model.eval, 1, X_np, derivativeId)
        end = time.time()
        res['MoCaX Cheb. eval'] = end - start
        res['MoCaX check'] = np.max(np.abs(y_mocax - values_np))
        # We also test random points...
        X_rn = a + np.random.uniform(size=(len(X_np), 8)) * (b-a)
        #
        start = time.time()
        V_rn = np.apply_along_axis(heston_function, 1, X_rn, None)
        res['Function_rn'] = time.time() - start
        #
        start = time.time()
        V_np = chebyshev_interpolation_np(X_rn, C_np, a, b)
        res['Numpy_rn'] = time.time() - start
        #
        start = time.time()
        V_mx = np.apply_along_axis(model.eval, 1, X_rn, derivativeId)
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
    import QuantLib as ql
    from heston_model_pricer import HestonModelPricer
    pricer = HestonModelPricer(integration=ql.AnalyticHestonEngine_Integration.gaussLaguerre(192))
    heston_function = lambda x, p : implied_volatility_from_vector(x, pricer)
    label_dict = {
        'term'            : ( 1/12, 5.0  ),
        'moneyness'       : ( -3.0, 3.0  ),
        'fwdPrice'        : ( 0.50, 1.50 ),
        'initial_vol'     : ( 0.10, 0.50 ),
        'long_vol_ratio'  : ( 0.50, 2.00 ),
        'decay_half_life' : ( 1.00, 5.00 ),
        'rho'             : (-0.80, 0.80 ),
        'feller_factor'   : ( 0.01, 4.00 ),
    }
    root_path = './'  # assume we start script from project root
    # we add some warm-up calculations at the beginning
    res = run_performance_testing([
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        ],
        label_dict, heston_function)
    res.to_csv(root_path + 'examples/heston/heston_mocax_performance.csv', sep=';')
    print(res)

