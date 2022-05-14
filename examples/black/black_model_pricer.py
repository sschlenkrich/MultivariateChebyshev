
import numpy as np
from scipy.stats import norm

def black_formula(x):
    """
    Normalised Black formula taking three inputs.

    In practice, callOrPut is either +1 or -1.
    For this example, we use callOrPut from [-1,1].
    """
    moneyness = x[0]
    stdDev    = x[1]
    callOrPut = x[2]
    d1 = np.log(moneyness) / stdDev + stdDev / 2.0
    d2 = d1 - stdDev
    return callOrPut * (moneyness*norm.cdf(callOrPut*d1)-norm.cdf(callOrPut*d2))
