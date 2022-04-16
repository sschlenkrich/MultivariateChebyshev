from cProfile import label
from cmath import exp
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import QuantLib as ql

sys.path.append("./")

from examples.heston_model_pricer import HestonModelPricer
from examples.heston_model_pricer import params_to_vector
from examples.heston_model_pricer import vector_to_params


class TestHestonModel(unittest.TestCase):
    """
    Test Heston model Vanilla option pricing
    """

    def test_heston_model_setup_and_pricing(self):
        # option inputs
        term = 1.0
        strikePrice = 1.0
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strikePrice)
        # model inputs
        riskFreeRate = 0.05
        dividYield = 0.01
        spotPrice = 1.0
        v0 = 0.005
        theta = 0.010
        kappa = 0.600
        sigma = 0.400  # a.k.a. xi
        rho = -0.15
        # QL structures
        today = ql.Settings.instance().evaluationDate
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, riskFreeRate, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, dividYield, ql.Actual365Fixed()))
        initialValue = ql.QuoteHandle(ql.SimpleQuote(spotPrice))
        hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, sigma, rho)
        hestonModel = ql.HestonModel(hestonProcess)
        # QL engine
        complexFormula = ql.AnalyticHestonEngine.AndersenPiterbargOptCV
        integration = ql.AnalyticHestonEngine_Integration.gaussLaguerre()
        engine = ql.AnalyticHestonEngine(hestonModel, complexFormula, integration)
        # QL low-level pricing
        riskFreeDiscount = riskFreeTS.discount(term)
        dividendDiscount = dividendTS.discount(term)
        value, evaluations = ql.AnalyticHestonEngine_doCalculation(
            riskFreeDiscount,
            dividendDiscount,
            spotPrice,
            strikePrice,
            term,
            kappa,
            theta,
            sigma,
            v0,
            rho,
            payoff,
            integration,
            complexFormula,
            engine,
            )
        # print(value)
        # print(evaluations)
        self.assertLess(abs(value - 0.04990652280565623), 1.0e-16)
        self.assertEqual(evaluations, 128)

    def test_heston_model_pricer(self):
        # option inputs
        term = 1.0
        strikePrice = 1.0
        call_or_put = 1.0
        # model inputs
        riskFreeRate = 0.05
        dividYield = 0.01
        spotPrice = 1.0
        v0 = 0.005
        theta = 0.010
        kappa = 0.600
        sigma = 0.400  # a.k.a. xi
        rho = -0.15
        #
        pricer = HestonModelPricer()
        value = pricer.option_price(
            np.exp(-riskFreeRate*term),
            np.exp(-dividYield*term),
            term,
            call_or_put,
            strikePrice,
            spotPrice,
            kappa,
            theta,
            sigma,
            v0,
            rho,
        )
        # print(value)
        self.assertLess(abs(value - 0.04990652280565623), 3.0e-16)

    def test_forward_pricer(self):
        # option inputs
        term = 1.0
        strikePrice = 1.0
        call_or_put = 1.0
        # model inputs
        #riskFreeRate = 0.05
        #dividYield = 0.01
        #spotPrice = 1.0
        v0 = 0.005
        theta = 0.010
        kappa = 0.600
        sigma = 0.400  # a.k.a. xi
        rho = -0.15
        #
        pricer = HestonModelPricer()
        f = lambda x : pricer.option_price(np.exp(-x[0]*term), np.exp(-x[1]*term), term, call_or_put,
                                           strikePrice, x[2], kappa, theta, sigma, v0, rho)
        #
        a = np.array([ -0.05, -0.05, 0.5, ])
        b = np.array([  0.05,  0.05, 2.0, ])
        np.random.seed(42)
        base2 = 10
        y = a + np.random.uniform(size=(2**base2, 3)) * (b-a)
        fwd_prices = np.apply_along_axis(f, axis=1, arr=y) / np.exp(-y[:,0]*term)
        z = np.array([ np.zeros(y.shape[0]), np.zeros(y.shape[0]), y[:,2] * np.exp(-y[:,1]*term) / np.exp(-y[:,0]*term) ]).T
        price_fwds = np.apply_along_axis(f, axis=1, arr=z)
        # print(np.max(np.abs(price_fwds - fwd_prices)))
        self.assertLess(np.max(np.abs(price_fwds - fwd_prices)), 5.0e-16)

    def test_implied_volatility(self):
        fwd_price = 1.0
        v0 = 0.040
        theta = 0.010
        kappa = 0.600
        sigma = 0.400  # a.k.a. xi
        rho = -0.15
        #
        pricer = HestonModelPricer()
        absTolerance = 1.0e-14
        maxEvaluations = 1000
        pricer.complexFormula = ql.AnalyticHestonEngine.OptimalCV
        # pricer.integration = ql.AnalyticHestonEngine_Integration.trapezoid(absTolerance, maxEvaluations)
        pricer.integration = ql.AnalyticHestonEngine_Integration.gaussLaguerre(192)
        f = lambda T, K : pricer.implied_volatility(T, K, fwd_price, v0, theta, kappa, rho, sigma)
        #
        terms = np.array([1/50, 0.25, 0.50, 1.00, 2.00])
        # plt.figure()
        for term in terms:
            stdev = np.sqrt((v0*np.exp(-kappa*term)+theta*(1-np.exp(-kappa*term)))*term)
            strikes = fwd_price * np.exp(np.linspace(-5*stdev, 5*stdev, 101))
            vols = np.array([ f(term, K) for K in strikes ])
            # plt.plot(strikes, vols, label='T=%.2f' % term)
        # plt.legend()
        # plt.show()

    def test_parameter_transformation(self):
        term = 2.0
        strikePrice = 1.2
        fwdPrice = 1.0
        v0 = 0.040
        theta = 0.010
        kappa = 0.600
        rho = -0.15
        sigma = 0.400  # a.k.a. xi
        #
        x = params_to_vector(term, strikePrice, fwdPrice, v0, theta, kappa, rho, sigma)
        p = vector_to_params(x)
        #
        self.assertEqual(x.shape, (8,))
        self.assertEqual(len(p), 8)
        self.assertAlmostEqual(p[0], term,        places = 16)
        self.assertAlmostEqual(p[1], strikePrice, places = 16)
        self.assertAlmostEqual(p[2], fwdPrice,    places = 16)
        self.assertAlmostEqual(p[3], v0,          places = 16)
        self.assertAlmostEqual(p[4], theta,       places = 16)
        self.assertAlmostEqual(p[5], kappa,       places = 16)
        self.assertAlmostEqual(p[6], rho,         places = 16)
        self.assertAlmostEqual(p[7], sigma,       places = 15)

    def test_volatility_calculation(self):
        label_dict = {
            'term'            : ( 1/12, 5.0  ),
            'moneyness'       : ( -4.0, 4.0  ),
            'fwdPrice'        : ( 0.50, 1.50 ),
            'initial_vol'     : ( 0.10, 0.50 ),
            'long_vol_ratio'  : ( 0.50, 2.00 ),
            'decay_half_life' : ( 1.00, 5.00 ),
            'rho'             : (-0.80, 0.80 ),
            'feller_factor'   : ( 0.01, 4.00 ),
        }
        a = np.array([ label_dict[k][0] for k in label_dict ])
        b = np.array([ label_dict[k][1] for k in label_dict ])
        np.random.seed(42)
        base2 = 10
        x = a + np.random.uniform(size=(2**base2, 8)) * (b-a)
        #
        p1 = HestonModelPricer()
        p1.complexFormula = ql.AnalyticHestonEngine.OptimalCV
        p1.integration = ql.AnalyticHestonEngine_Integration.gaussLaguerre(192)
        f1 = lambda x : p1.implied_volatility(*vector_to_params(x))
        #
        p2 = HestonModelPricer()
        absTolerance = 1.0e-12
        maxEvaluations = 100
        p2.integration = ql.AnalyticHestonEngine_Integration.trapezoid(absTolerance, maxEvaluations)
        f2 = lambda x : p2.implied_volatility(*vector_to_params(x))
        #
        y1 = np.apply_along_axis(f1, axis=1, arr=x)
        y2 = np.apply_along_axis(f2, axis=1, arr=x)
        # print(np.max(np.abs(y1 - y2)))
        self.assertLess(np.max(np.abs(y1 - y2)), 0.02)



if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestHestonModel))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
