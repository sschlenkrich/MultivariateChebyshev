

import multiprocessing
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

try:
    from tf_config import tensorflow as tf
    # control tf version via tf_config if necessary
except ImportError:
    pass
from src.multivariate_chebyshev_tensorflow import chebyshev_multi_points  as chebyshev_multi_points_tf
from src.multivariate_chebyshev_tensorflow import chebyshev_transform     as chebyshev_transform_tf
from src.multivariate_chebyshev_tensorflow import chebyshev_coefficients  as chebyshev_coefficients_tf
from src.multivariate_chebyshev_tensorflow import chebyshev_interpolation as chebyshev_interpolation_tf

from julia.api import Julia
jl_instance = Julia(compiled_modules=False)  # avoid Julia and PyJulia setup error.
from julia import Main as jl

def run_performance_testing(perf_degrees, label_dict, heston_model_pricer, root_path):
    """
    Run the performance testing and return a Dataframe that contains the run times.

    perf_degrees ... a list of Chebyshev polynomial degrees.

    label_dict ... a dictionary of parameters and boundaries

    heston_model_pricer ... a HestonModelPricer object
    """
    jl.include(root_path + 'src/multivariate_chebyshev_julia.jl')  # double-check ./ versus ../
    #
    print('Use label_dict = ')
    pprint(label_dict)
    a = np.array([label_dict[k][0] for k in label_dict])
    b = np.array([label_dict[k][1] for k in label_dict])
    np.random.seed(42)
    v_shifts = np.random.uniform(size=len(perf_degrees))  # random shifts to avoid optimisation under the hood
    results = []
    for Nd, eps in zip(perf_degrees, v_shifts):
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
        values_np  = np.apply_along_axis(implied_volatility_from_vector, 1, X_np, heston_model_pricer)
        end = time.time()
        values_np += eps
        res['Function_np'] = end - start
        #
        start = time.time()
        C_np = chebyshev_coefficients_np(degrees, Z_np, values_np)
        end = time.time()
        res['Numpy'] = end - start
        #
        print('  Run Tensorflow calculations.')
        Z_tf = chebyshev_multi_points_tf(degrees)
        X_tf = chebyshev_transform_tf(Z_tf, a, b)
        #
        start = time.time()
        C_tf = chebyshev_coefficients_tf(degrees, Z_tf, values_np)
        end = time.time()
        res['Tensorflow'] = end - start
        #
        print('  Run Julia calculations.')
        Z_jl = jl.chebyshev_multi_points(degrees)
        X_jl = jl.chebyshev_transform(Z_jl, a.reshape((1,-1)), b.reshape((1,-1)))
        #
        start = time.time()
        values_jl = np.apply_along_axis(implied_volatility_from_vector, 1, X_jl, heston_model_pricer)  # different ordering
        end = time.time()
        # we save the values for Julia script and skip the rest
        filename = root_path + 'examples/values_jl_%d.csv' % Nd
        np.savetxt(filename, values_jl)
        #
        values_jl += eps
        res['Function_jl'] = end - start
        #
        start = time.time()
        C_jl_bm = jl.chebyshev_coefficients(degrees, Z_jl, values_jl)
        end = time.time()
        res['Julia (batchmul)'] = end - start
        #
        start = time.time()
        C_jl_mm = jl.chebyshev_coefficients(degrees, Z_jl, values_jl, jl.matmul)
        end = time.time()
        res['Julia (matmul)'] = end - start
        #
        print('  |C_tf - C_np|:     %.2e' % np.max(np.abs(C_tf - C_np)))
        print('  |C_jl_bm - C_np|:  %.2e' % np.max(np.abs(C_jl_bm - C_np)))
        print('  |C_jl_mm - C_np|:  %.2e' % np.max(np.abs(C_jl_mm - C_np)))
        assert np.max(np.abs(C_tf - C_np)) < 1.0e-6
        assert np.max(np.abs(C_jl_bm - C_np)) < 1.0e-14
        assert np.max(np.abs(C_jl_mm - C_np)) < 1.0e-14
        results.append(res)
    results = pd.DataFrame(results)
    print('Done.')
    return results




if __name__ == '__main__':
    import QuantLib as ql
    from heston_model_pricer import HestonModelPricer
    pricer = HestonModelPricer(integration=ql.AnalyticHestonEngine_Integration.gaussLaguerre(192))
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
    res = run_performance_testing([1, 1, 1, 2], label_dict, pricer, root_path)
    res.to_csv('./examples/heston_chebyshev_performance.csv', sep=';')
    print(res)
