
import sys
import unittest

import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

sys.path.append("./")

from src.multivariate_chebyshev_tensorflow import cartesian_product
from src.multivariate_chebyshev_tensorflow import chebyshev_multi_points
from src.multivariate_chebyshev_tensorflow import chebyshev_transform
from src.multivariate_chebyshev_tensorflow import chebyshev_coefficients
from src.multivariate_chebyshev_tensorflow import chebyshev_interpolation

class TestCartesianProductTensorflow(unittest.TestCase):
    """
    Test cartesian product calculation
    """

    def test_indexing(self):
        x = tf.constant([1, 2])
        y = tf.constant([1, 2, 3])
        z = tf.constant([1, 2, 3, 4])
        p = cartesian_product(x, y, z)  # call with argument list
        p = cartesian_product(*[x, y, z]) # call with list and unpack
        v = 100*p[:,0] + 10*p[:,1] + p[:,2]
        V = tf.reshape(v, (x.shape[0],y.shape[0],z.shape[0]))
        V_ref = tf.constant([
            [ [111, 112, 113, 114],
              [121, 122, 123, 124],
              [131, 132, 133, 134] ],       
            [ [211, 212, 213, 214],
              [221, 222, 223, 224],
              [231, 232, 233, 234] ]
        ])
        self.assertEqual(float(tf.math.reduce_max(tf.abs(V-V_ref))), 0.0)


class TestBlackFormulaTensorflow(unittest.TestCase):
    """
    Test cartesian product calculation
    """

    @staticmethod
    def BlackOverK(x):
        moneyness, stdDev, callOrPut = x
        d1 = tf.math.log(moneyness) / stdDev + stdDev / 2.0
        d2 = d1 - stdDev
        return callOrPut * (moneyness*norm.cdf(callOrPut*d1)-norm.cdf(callOrPut*d2))

    def test_black_formula_chebyshev_points(self):
        a = tf.constant([ 0.5, 0.01, -1.0 ])
        b = tf.constant([ 2.0, 0.50, +1.0 ])
        degrees = [ 3, 4, 5 ]
        #
        multi_points = chebyshev_multi_points(degrees)
        Y = chebyshev_transform(multi_points, a, b)
        values = tf.map_fn(TestBlackFormulaTensorflow.BlackOverK, Y)
        C = chebyshev_coefficients(degrees, multi_points, values)
        #
        z = chebyshev_interpolation(Y, C, a, b)
        self.assertLess(float(tf.math.reduce_max(tf.abs(z - values))), 4.0e-7)

    def test_black_formula_random_points(self):
        a = tf.constant([ 0.5, 0.50, -1.0 ])
        b = tf.constant([ 2.0, 2.50, +1.0 ])
        degrees = [ 5, 5, 5 ]
        #
        multi_points = chebyshev_multi_points(degrees)
        Y = chebyshev_transform(multi_points, a, b)
        values = tf.map_fn(TestBlackFormulaTensorflow.BlackOverK, Y)
        C = chebyshev_coefficients(degrees, multi_points, values)
        #
        tf.random.set_seed(42)
        base2 = 10
        y = a + tf.random.uniform(shape=(2**base2, 3)) * (b-a)
        z = chebyshev_interpolation(y, C, a, b)
        z_ref = tf.map_fn(TestBlackFormulaTensorflow.BlackOverK, y)
        self.assertLess(float(tf.math.reduce_max(tf.abs(z - z_ref))), 5.0e-3)




if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCartesianProductTensorflow))
    suite.addTest(unittest.makeSuite(TestBlackFormulaTensorflow))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
