

import numpy as np
import pandas as pd
from pprint import pprint
import sys
import time

sys.path.append('./')

from black_model_pricer import black_formula

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

try:
    import mocaxpy as mx
except ImportError:
    mx = None

def run_performance_testing(perf_degrees, root_path):
    """
    Run the performance testing and return a Dataframe that contains the run times.

    perf_degrees ... a list of Chebyshev polynomial degrees.

    """
    jl.include(root_path + 'src/multivariate_chebyshev_julia.jl')  # double-check ./ versus ../
    #
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
        # We add testing with random point interpolation
        X_rn = a + np.random.uniform(size=(n_points, 3)) * (b-a)
        print('Run Nd = %d with %d Chebyshev points...' % (Nd,n_points))
        res = { 'Nd' : Nd, 'N_Points' : n_points }
        #
        print('  Run Numpy calculations.')
        Z_np = chebyshev_multi_points_np(degrees)
        X_np = chebyshev_transform_np(Z_np, a, b)
        #
        start = time.time()
        values_np  = np.apply_along_axis(black_formula, 1, X_np)
        end = time.time()
        res['Function_np'] = end - start
        #
        start = time.time()
        C_np = chebyshev_coefficients_np(degrees, Z_np, values_np)
        end = time.time()
        res['Numpy_coeff'] = end - start
        #
        start = time.time()
        V_np = chebyshev_interpolation_np(X_rn, C_np, a, b)
        end = time.time()
        res['Numpy_interp'] = end - start
        #
        print('  Run Tensorflow calculations.')
        Z_tf = chebyshev_multi_points_tf(degrees)
        X_tf = chebyshev_transform_tf(Z_tf, a, b)
        #
        start = time.time()
        C_tf = chebyshev_coefficients_tf(degrees, Z_tf, values_np)
        end = time.time()
        res['Tensorflow_coeff'] = end - start
        #
        start = time.time()
        V_tf = chebyshev_interpolation_tf(tf.cast(X_rn, tf.float32), C_tf, a, b)
        end = time.time()
        res['Tensorflow_interp'] = end - start
        #
        print('  Run Julia calculations.')
        Z_jl = jl.chebyshev_multi_points(degrees)
        X_jl = jl.chebyshev_transform(Z_jl, a.reshape((1,-1)), b.reshape((1,-1)))
        #
        start = time.time()
        values_jl = np.apply_along_axis(black_formula, 1, X_jl)  # different ordering
        end = time.time()
        #
        res['Function_jl'] = end - start
        #
        start = time.time()
        C_jl_bm = jl.chebyshev_coefficients(degrees, Z_jl, values_jl)
        end = time.time()
        res['Julia_coeff (batchmul)'] = end - start
        #
        start = time.time()
        V_jl_bm = jl.chebyshev_interpolation(X_rn, C_jl_bm, a.reshape((1,-1)), b.reshape((1,-1)))
        end = time.time()
        res['Julia_interp (batchmul)'] = end - start
        #
        start = time.time()
        C_jl_mm = jl.chebyshev_coefficients(degrees, Z_jl, values_jl, jl.matmul)
        end = time.time()
        res['Julia_coeff (matmul)'] = end - start
        #
        start = time.time()
        V_jl_mm = jl.chebyshev_interpolation(X_rn, C_jl_mm, a.reshape((1,-1)), b.reshape((1,-1)), jl.matmul)
        end = time.time()
        res['Julia_interp (matmul)'] = end - start
        #
        print('  |C_tf - C_np|:     %.2e' % np.max(np.abs(C_tf - C_np)))
        print('  |C_jl_bm - C_np|:  %.2e' % np.max(np.abs(C_jl_bm - C_np)))
        print('  |C_jl_mm - C_np|:  %.2e' % np.max(np.abs(C_jl_mm - C_np)))
        #
        print('  |V_tf - V_np|:     %.2e' % np.max(np.abs(V_tf - V_np)))
        print('  |V_jl_bm - V_np|:  %.2e' % np.max(np.abs(V_jl_bm - V_np)))
        print('  |V_jl_mm - V_np|:  %.2e' % np.max(np.abs(V_jl_mm - V_np)))
        #
        assert np.max(np.abs(C_tf - C_np)) < 5.0e-6
        assert np.max(np.abs(C_jl_bm - C_np)) < 1.0e-14
        assert np.max(np.abs(C_jl_mm - C_np)) < 1.0e-14
        #
        assert np.max(np.abs(V_tf - V_np)) < 5.0e-5
        assert np.max(np.abs(V_jl_bm - V_np)) < 1.0e-14
        assert np.max(np.abs(V_jl_mm - V_np)) < 1.0e-14
        if mx is None:  # skip remainder of the iteration
            results.append(res)
            continue
        #
        print('  Run MoCaX calculations.')
        black_formula_mocax = lambda x, p : black_formula(x)
        start = time.time()
        model = mx.Mocax(
            black_formula_mocax,
            len(degrees),
            mx.MocaxDomain([[a_,b_] for a_, b_ in zip(a,b)]),
            None,
            mx.MocaxNs(degrees),
            max_derivative_order=0,
        )
        end = time.time()
        res['MoCaX_construct'] = end - start
        derivativeId = model.get_derivative_id([0]*3)
        #
        start = time.time()
        V_mx = np.apply_along_axis(model.eval, 1, X_rn, derivativeId)
        end = time.time()
        res['MoCaX_interp'] = end - start
        print('  |V_mx - V_np|:  %.2e' % np.max(np.abs(V_mx - V_np)))
        assert np.max(np.abs(V_mx - V_np)) < 1.0e-14
        #
        results.append(res)
    results = pd.DataFrame(results)
    print('Done.')
    return results




if __name__ == '__main__':
    root_path = './'  # assume we start script from project root
    # we add some warm-up calculations at the beginning
    res = run_performance_testing([
        10, 10, 10,
        20, 20, 20,
        40, 40, 40,
        ],root_path)
    res.to_csv('./examples/black/black_chebyshev_performance.csv', sep=',')
    print(res)
