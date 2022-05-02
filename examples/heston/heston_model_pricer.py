
import numpy as np
import QuantLib as ql

class HestonModelPricer:
    """
    A wrapper for QuantLib's AnalyticHestonEngine
    using low-level pricing functions.
    """

    def __init__(self, complexFormula=None, integration=None):
        # QL term structures and quotes are required for process construction.
        # But these inputs are not used in low-level functions.
        self.today = ql.Settings.instance().evaluationDate
        # Complex formula methodology and integration method are critical
        # for numerical accuracy.
        # For a discussion of different methods see
        # https://hpcquantlib.wordpress.com/2017/05/07/newer-semi-analytic-heston-pricing-algorithms/
        self.complexFormula = ql.AnalyticHestonEngine.AndersenPiterbargOptCV
        if complexFormula is not None:
            self.complexFormula = complexFormula
        self.integration = ql.AnalyticHestonEngine_Integration.gaussLaguerre()
        if integration is not None:
            self.integration = integration


    def option_price(self,
                     riskFreeDiscount,
                     dividendDiscount,
                     term,
                     call_or_put,
                     strikePrice,
                     spotPrice,
                     kappa,
                     theta,
                     xi,  # sigma in QantLib's notation
                     v0,
                     rho,
                    ):
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(self.today, -np.log(riskFreeDiscount)/term, ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(self.today, -np.log(dividendDiscount)/term, ql.Actual365Fixed()))
        hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, ql.QuoteHandle(ql.SimpleQuote(spotPrice)), v0, kappa, theta, xi, rho)
        hestonModel = ql.HestonModel(hestonProcess)
        engine = ql.AnalyticHestonEngine(hestonModel, self.complexFormula, self.integration)
        if call_or_put == 1.0:
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strikePrice)
        elif call_or_put == -1.0:
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, strikePrice)
        else:
            raise ValueError('call_or_put must be 1 or -1. Got ' + str(call_or_put) + '.')
        if hasattr(ql, 'AnalyticHestonEngine_doCalculation'):
            value, evaluations = ql.AnalyticHestonEngine_doCalculation(
                riskFreeDiscount,
                dividendDiscount,
                spotPrice,
                strikePrice,
                term,
                kappa,
                theta,
                xi,
                v0,
                rho,
                payoff,
                self.integration,
                self.complexFormula,
                engine,
                )
        else:
            # use standard QuantLib methodology
            exerciseDate = self.today + round(365 * term)
            exercise = ql.EuropeanExercise(exerciseDate)
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(engine)
            value = option.NPV()
        return value

    def implied_volatility(self,
                           term,
                           strikePrice,
                           fwdPrice,
                           v0,          # short term variance
                           theta,       # long term variance
                           kappa,       # term structure
                           rho,         # skew
                           xi,          # smile
                           ):
        call_or_put = 1 if strikePrice > fwdPrice else -1
        black_price = self.option_price(1.0, 1.0, term, call_or_put, strikePrice,
                                      fwdPrice, kappa, theta, xi, v0, rho)
        option_type = ql.Option.Call if call_or_put == 1 else ql.Option.Put
        vol = ql.blackFormulaImpliedStdDev(option_type, strikePrice, fwdPrice, black_price) / np.sqrt(term)
        return vol

def heston_stdev(v0, theta, kappa, term):
    return np.sqrt((v0*np.exp(-kappa*term)+theta*(1-np.exp(-kappa*term)))*term)


def params_to_vector(
    term,
    strikePrice,
    fwdPrice,
    v0,          # short term variance
    theta,       # long term variance
    kappa,       # term structure
    rho,         # skew
    xi,          # smile
    ):
    """
    Transform original vol inputs to normalised input vector 
    """
    x = np.zeros(8)
    x[0] = term
    x[1] = np.log(strikePrice / fwdPrice) / heston_stdev(v0, theta, kappa, term)  # moneyness
    x[2] = fwdPrice
    x[3] = np.sqrt(v0)        # initial volatility
    x[4] = np.sqrt(theta/v0)  # long-term volatility ratio
    x[5] = 0.7 / kappa        # decay half life, from exp(-at)=1/2
    x[6] = rho
    x[7] = xi**2 / (2 * kappa * theta)  # Feller factor
    return x

def vector_to_params(x):
    """
    Transform normalised input to original parameters
    """
    assert x.shape == (8,)
    term     = x[0]
    fwdPrice = x[2]
    v0       = x[3]**2
    theta    = x[4]**2 * v0
    kappa    = 0.7 / x[5]
    rho      = x[6]
    xi       = np.sqrt(2 * kappa * theta * x[7])
    strikePrice = np.exp(x[1] * heston_stdev(v0, theta, kappa, term)) * fwdPrice
    return (term, strikePrice, fwdPrice, v0, theta, kappa, rho, xi)


def implied_volatility_from_vector(x, heston_model_pricer):
    """
    Wrap implied volatility calculation into a function.
    """
    return heston_model_pricer.implied_volatility(*vector_to_params(x))
