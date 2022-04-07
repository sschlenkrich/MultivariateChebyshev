
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.append("./")

from src.multivariate_chebyshev_numpy import cartesian_product
from src.multivariate_chebyshev_numpy import chebyshev_multi_points
from src.multivariate_chebyshev_numpy import chebyshev_transform
from src.multivariate_chebyshev_numpy import chebyshev_coefficients
from src.multivariate_chebyshev_numpy import chebyshev_interpolation

class TestCartesianProductNumpy(unittest.TestCase):
    """
    Test cartesian product calculation
    """

    def test_indexing(self):
        x = np.array([1, 2])
        y = np.array([1, 2, 3])
        z = np.array([1, 2, 3, 4])
        p = cartesian_product(x, y, z)  # call with argument list
        p = cartesian_product(*[x, y, z]) # call with list and unpack
        print('')
        print(p)
        v = np.array([ 100*x[0]+10*x[1]+x[2] for x in p ])
        V = v.reshape((len(x),len(y),len(z)))
        V_ref = np.array([
            [ [111, 112, 113, 114],
              [121, 122, 123, 124],
              [131, 132, 133, 134] ],       
            [ [211, 212, 213, 214],
              [221, 222, 223, 224],
              [231, 232, 233, 234] ]
        ])
        self.assertEqual(np.max(np.abs(V-V_ref)), 0.0)


class TestBlackFormulaNumpy(unittest.TestCase):
    """
    Test cartesian product calculation
    """

    @staticmethod
    def BlackOverK(x):
        moneyness, stdDev, callOrPut = x
        d1 = np.log(moneyness) / stdDev + stdDev / 2.0
        d2 = d1 - stdDev
        return callOrPut * (moneyness*norm.cdf(callOrPut*d1)-norm.cdf(callOrPut*d2))

    def test_black_formula_chebyshev_points(self):
        a = np.array([ 0.5, 0.01, -1.0 ])
        b = np.array([ 2.0, 0.50, +1.0 ])
        degrees = [ 3, 4, 5 ]
        #
        multi_points = chebyshev_multi_points(degrees)
        Y = chebyshev_transform(multi_points, a, b)
        values = np.apply_along_axis(TestBlackFormulaNumpy.BlackOverK, 1, Y)
        C = chebyshev_coefficients(degrees, multi_points, values)
        #
        z = chebyshev_interpolation(Y, C, a, b)
        self.assertLess(np.max(np.abs(z - values)), 5.0e-16)

    def test_black_formula_random_points(self):
        a = np.array([ 0.5, 0.50, -1.0 ])
        b = np.array([ 2.0, 2.50, +1.0 ])
        degrees = [ 5, 5, 5 ]
        #
        multi_points = chebyshev_multi_points(degrees)
        Y = chebyshev_transform(multi_points, a, b)
        values = np.apply_along_axis(TestBlackFormulaNumpy.BlackOverK, 1, Y)
        C = chebyshev_coefficients(degrees, multi_points, values)
        #
        np.random.seed(42)
        base2 = 13
        y = a + np.random.uniform(size=(2**base2, 3)) * (b-a)
        z = chebyshev_interpolation(y, C, a, b)
        z_ref = np.apply_along_axis(TestBlackFormulaNumpy.BlackOverK, 1, y)
        self.assertLess(np.max(np.abs(z - z_ref)), 7.0e-3)

    @unittest.skip('Only for ad-hoc testing.')
    def test_plot_black_formula_interpolation(self):
        a = np.array([ 0.5, 0.01, -1.0 ])
        b = np.array([ 2.0, 0.50, +1.0 ])
        degrees = [ 10, 10, 10 ]
        #
        multi_points = chebyshev_multi_points(degrees)
        Y = chebyshev_transform(multi_points, a, b)
        values = np.apply_along_axis(TestBlackFormulaNumpy.BlackOverK, 1, Y)
        C = chebyshev_coefficients(degrees, multi_points, values)
        #
        callOrPut = 1.0
        moneyness = np.linspace(0.5, 2.0, 101)
        fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
        for stDev, ax in zip([ 0.01, 0.10, 0.25, 0.50 ], axs.reshape(-1)):
            v = lambda x : np.array([x, stDev, callOrPut])
            T = lambda x : chebyshev_interpolation(np.array([x]), C, a, b)[0]
            ax.plot(moneyness, [ TestBlackFormulaNumpy.BlackOverK(v(x)) for x in moneyness], 'r', label='Black')
            ax.plot(moneyness, [ T(v(x)) for x in moneyness], 'b', label='Chebyshev')
            ax.set_title('stDev = %.2f' % stDev)
        axs.reshape(-1)[0].legend()
        axs[1,0].set_xlabel('moneyness')
        axs[1,1].set_xlabel('moneyness')
        axs[0,0].set_ylabel('forward price')
        axs[1,0].set_ylabel('forward price')
        fig.set_figheight(8)
        fig.set_figwidth(12)
        fig.suptitle('Call/Put = %.1f' % callOrPut)
        plt.tight_layout()
        plt.show()        



if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCartesianProductNumpy))
    suite.addTest(unittest.makeSuite(TestBlackFormulaNumpy))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
