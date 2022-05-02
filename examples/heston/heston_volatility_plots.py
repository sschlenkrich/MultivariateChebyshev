
import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from heston_model_pricer import heston_stdev
from heston_model_pricer import vector_to_params


def plot_smiles(terms, moneyness, fwdPrice, initial_vol, long_vol_ratio, decay_half_life, rho, feller_factor,
                f_model, f_proxy = None):
    # We specify the functions to plot
    sigma = lambda T, M : f_model(np.array([
        T, M, fwdPrice, initial_vol, long_vol_ratio, decay_half_life, rho, feller_factor
        ]))
    sigma_proxy = None   
    if f_proxy is not None:
        sigma_proxy = lambda T, M : f_proxy(np.array([
            T, M, fwdPrice, initial_vol, long_vol_ratio, decay_half_life, rho, feller_factor
            ]))
    term, strikePrice, fwdPrice, v0, theta, kappa, rho, xi = vector_to_params(
        np.array([ 0.0, 0.0, fwdPrice, initial_vol, long_vol_ratio, decay_half_life, rho, feller_factor])
    )
    # We want colors from a color map
    color_map = plt.get_cmap('jet')
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(terms))
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=color_map)
    # We also want to set some figure properties
    fig = plt.figure(figsize=(10, 8))
    for idx, T in enumerate(terms):
        m = np.linspace(moneyness[0],moneyness[1], 101)        
        stdev = heston_stdev(v0, theta, kappa, T)
        strikes = fwdPrice * np.exp(stdev * m)
        #
        colorVal = scalarMap.to_rgba(idx)
        plt.plot(strikes, np.array([ sigma(T, m_) for m_ in m]), '-', color=colorVal, label='T=%.2f' % T)
        if sigma_proxy is not None:
            plt.plot(strikes, np.array([ sigma_proxy(T, m_) for m_ in m]), '.', color=colorVal)
        plt.xlabel('strike price')
        plt.ylabel('implied volatility')
    plt.title(r'$v_0=%.4f$, $\theta=%.4f$, $\kappa=%0.4f$, $\rho=%.4f$, $\xi=%.4f$' % (v0,theta,kappa,rho,xi))
    plt.legend()


def get_widgets(terms, label_dict, f_model, f_proxy):
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
        'f_model'         : widgets.fixed(f_model),
        'f_proxy'         : widgets.fixed(f_proxy),
    }
    return kwargs
