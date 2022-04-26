
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from heston_model_pricer import HestonModelPricer
from heston_model_pricer import heston_stdev
from heston_model_pricer import vector_to_params


def plot_smiles(terms, moneyness, fwdPrice, initial_vol, long_vol_ratio, decay_half_life, rho, feller_factor):
    pricer = HestonModelPricer()
    sigma = lambda T, M : pricer.implied_volatility(*vector_to_params(np.array([
        T, M, fwdPrice, initial_vol, long_vol_ratio, decay_half_life, rho, feller_factor
        ])))
    term, strikePrice, fwdPrice, v0, theta, kappa, rho, xi = vector_to_params(
        np.array([ 0.0, 0.0, fwdPrice, initial_vol, long_vol_ratio, decay_half_life, rho, feller_factor])
    )
    for T in terms:
        m = np.linspace(moneyness[0],moneyness[1], 101)        
        stdev = heston_stdev(v0, theta, kappa, T)
        strikes = fwdPrice * np.exp(stdev * m)
        v = np.array([ sigma(T, m_) for m_ in m])
        plt.plot(strikes, v, label='T=%.2f' % T)
        plt.xlabel('strike price')
        plt.ylabel('implied volatility')
    plt.legend()


def get_widgets(terms, label_dict):
    d = label_dict
    kwargs = {
        'terms'           : widgets.fixed(terms),
        'moneyness'       : widgets.FloatRangeSlider(value=d['moneyness'],                           min=2*d['moneyness'][0],     max=2.0*d['moneyness'][1]  ),
        'fwdPrice'        : widgets.FloatSlider((d['fwdPrice'][0]       +d['fwdPrice'][1])/2,        min=d['fwdPrice'][0],        max=d['fwdPrice'][1]       ),
        'initial_vol'     : widgets.FloatSlider((d['initial_vol'][0]    +d['initial_vol'][1])/2,     min=d['initial_vol'][0],     max=d['initial_vol'][1]    ),
        'long_vol_ratio'  : widgets.FloatSlider((d['long_vol_ratio'][0] +d['long_vol_ratio'][1])/2,  min=d['long_vol_ratio'][0],  max=d['long_vol_ratio'][1] ),
        'decay_half_life' : widgets.FloatSlider((d['decay_half_life'][0]+d['decay_half_life'][1])/2, min=d['decay_half_life'][0], max=d['decay_half_life'][1]),
        'rho'             : widgets.FloatSlider((d['rho'][0]            +d['rho'][1])/2,             min=d['rho'][0],             max=d['rho'][1]            ),
        'feller_factor'   : widgets.FloatSlider((d['feller_factor'][0]  +d['feller_factor'][1])/2,   min=d['feller_factor'][0],   max=d['feller_factor'][1]  ),
    }
    return kwargs
