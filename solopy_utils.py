#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 14:29:12 2026

@author: shart
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

def ma_front(x, window, axis=1, ignore_nan=False):
    """
    Trailing-edge moving average with correct normalization:
    output[i] = mean(x[i : i+window]) where the window at the end
    is shorter and averaged only over available samples.
    """
    weights = np.ones(window)
    wrev = weights[::-1]

    def _ma(m):
        if ignore_nan:
            # Replace NaNs with 0 for summation
            m0 = np.where(np.isnan(m), 0, m)
            conv_vals = np.convolve(m0, wrev, mode='full')
            # Count non-NaN elements in each window
            mask = (~np.isnan(m)).astype(float)
            conv_norm = np.convolve(mask, wrev, mode='full')
            conv_norm[conv_norm == 0] = np.nan  # avoid division by zero
        else:
            conv_vals = np.convolve(m, wrev, mode='full')
            conv_norm = np.convolve(np.ones_like(m), wrev, mode='full')

        vals = conv_vals[window-1 : window-1 + m.size]
        norm = conv_norm[window-1 : window-1 + m.size]
        return vals / norm
    return np.apply_along_axis(_ma, axis, x)

def ma_cent(x, window, axis=1):
    """
    Centered moving average with correct normalization:
    output[i] = mean(x[i - (window-1)//2 : i + (window+1)//2]),
    with shorter windows at the edges.
    """
    weights = np.ones(window)
    wrev = weights[::-1]
    def _ma(m):
        conv_vals = np.convolve(m, wrev, mode='full')
        conv_norm = np.convolve(np.ones_like(m), wrev, mode='full')

        half = window // 2
        vals = conv_vals[half : half + m.size]
        norm = conv_norm[half : half + m.size]
        return vals / norm
    return np.apply_along_axis(_ma, axis, x)

#-----Conversion functions----#
def oos_to_epn(inv_v_auhr):
    AU = 1.495978707e11       # meters
    HR = 3600.0               # seconds
    m_p = 1.6726219e-27       # kg (proton)
    q_e = 1.602176634e-19     # C
    """Convert 1/(AU/hr) to energy in keV"""
    v_m_s = 1.0 / inv_v_auhr * (AU / HR)
    E_joules = 0.5 * m_p * v_m_s**2
    return E_joules / q_e / 1e6  # in keV

def epn_to_oos(E_mev):
    AU = 1.495978707e11       # meters
    HR = 3600.0               # seconds
    m_p = 1.6726219e-27       # kg (proton)
    q_e = 1.602176634e-19     # C
    """Convert energy in keV to 1/(AU/hr)"""
    E_joules = E_mev * 1e6 * q_e
    v_m_s = np.sqrt(2 * E_joules / m_p)
    v_auhr = v_m_s * (HR / AU)
    return 1.0 / (v_auhr+0.001)  # inverse speed in AU/hr
#-----------------------------#

#--------------Plot Manipulation---------------#
def multiple_locator(ax,start,stop):
    span = (stop - start).total_seconds()/86400
    if span < 0.5:
        major, minor = 1/24, 1/48
    elif 0.5 <= span < 1:
        major, minor = 1/8, 1/48
    elif 1 <= span < 3:
        major, minor = 1/3, 1/24
    elif 3 <= span < 10:
        major, minor = 1, 1/24
    elif 10 <= span < 20:
        major, minor = 3, 1/8
    elif 20 <= span < 60:
        major, minor = 7, 1/2
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        
    if span < 60:
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
        
    if span < 1:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b. %d\n%H:%M'))
    elif span < 7:
        last_date = [None]
        def time_formatter(x, pos=None):
            dt = mdates.num2date(x)
            if dt.date() != last_date[0]:
                last_date[0] = dt.date()
                return dt.strftime('%b. %d\n%H:%M').replace(' 0', ' ')
            return dt.strftime('%H:%M')

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    # elif 3 <= span < 60:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%b. %d'))

    
def tick_parameters(ax,labelbottom=False,color='black',labelcolor='black',right=True,left=True):
    ax.tick_params(which="major", top=True, bottom=True, right=right, left=left, length=6, width=0.75,\
                   direction="inout",labeltop=False,labelbottom=labelbottom,labelsize=10,color=color,\
                   labelcolor=labelcolor)
    ax.tick_params(which="minor", top=True, bottom=True, right=right, left=left, length=3, width=0.75, \
                   direction="inout",labeltop=False,labelbottom=False,labelsize=10,color=color,\
                   labelcolor=labelcolor)   
        
def curate_axes(axs,start,stop,year=None):
    axs = np.atleast_1d(axs)
    for ax in axs:
        ax.set_xlim(start,stop)
        multiple_locator(ax,start,stop)
        tick_parameters(ax)
        if ax.get_yscale() == 'log':
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        if ax == axs[-1]:
            tick_parameters(ax,labelbottom=True)
            if year != None:
                ax.set_xlabel('Date (%d in UTC)'%year)
            else:
                ax.set_xlabel('Date (UTC)')

def pspiral_length(r_au, vsw=400):
    AU = 1.496e8  # km
    OMEGA_SUN = 2.865e-6  # rad/s
    r = r_au * AU
    a = OMEGA_SUN / vsw
    term1 = 0.5 * r * np.sqrt(1 + (a*r)**2)
    term2 = np.arcsinh(a*r) / (2*a)
    length_km = term1 + term2
    return length_km / AU

def dispersion_plot(ax, dist, time, vsw=400, erange=[0.04, 12], space='energy', color='black', ls='-', lw=1):
    AU = 1.496e8  # km
    c = 299792.0 # km/s
    
    L = pspiral_length(dist, vsw=vsw)
    E = np.linspace(erange[0], erange[1], 30000)
    
    gamma = 1 + E/938.272
    beta = np.sqrt(1 - 1/(gamma**2))
    v = beta * c
    v_auhr = v * 3600/AU
    travel_hr = L / v_auhr
    arrival = time + pd.to_timedelta(travel_hr, unit='h')
    
    if space == 'energy': y = E
    elif space == 'inverse_v': y = 1/v_auhr
    ax.plot(arrival, y, lw=lw, color=color,ls=ls) 
    return L 
#----------------------------------------------#

def get_solo_species():
    return np.array(['H','He3','He4','C','N','O','Ne','Mg','Si','S','Ca','Fe'])

def species_utils(species):
    util = {}
    if species == 'H':
        util['color'] = 'black'
        util['name'] = 'H'
        util['erange'] = [0.23, 8.3]
    if species == 'He3':
        util['color'] = 'red'
        util['name'] = '$^3$He'
        util['erange'] = [0.168, 10]
    if species == 'He4':
        util['color'] = 'blue'
        util['name'] = '$^4$He'
        util['erange'] = [0.157, 9.4]
    if species == 'C':
        util['color'] = 'purple'
        util['name'] = 'C'
        util['erange'] = [0.063, 9.4]
    if species == 'N':
        util['color'] = 'turquoise'
        util['name'] = 'N'
        util['erange'] = [0.054, 9.4]
    if species == 'O':
        util['color'] = 'orange'
        util['name'] = 'O'
        util['erange'] = [0.054, 9.4]
    if species == 'Ne':
        util['color'] = 'gray'
        util['name'] = 'Ne'
        util['erange'] = [0.54, 9.4]
    if species == 'Mg':
        util['color'] = 'brown'
        util['name'] = 'Mg'
        util['erange'] = [0.054, 9.4]
    if species == 'Si':
        util['color'] = 'pink'
        util['name'] = 'Si'
        util['erange'] = [0.054, 9.4]
    if species == 'S':
        util['color'] = 'cyan'
        util['name'] = 'S'
        util['erange'] = [0.054, 9.4]
    if species == 'Ca':
        util['color'] = 'gold'
        util['name'] = 'Ca'
        util['erange'] = [0.054, 9.4]
    if species == 'Fe':
        util['color'] = 'green'
        util['name'] = 'Fe'
        util['erange'] = [0.036, 10]
    if species == 'Heavy':
        util['color'] = 'orange'
        util['name'] = 'C-Fe'
        util['erange'] = [0.01, 10]
    return util

def monochrome_colors(base_color, n, min_intensity=0, max_lightness=0):
    rgb = np.array(mcolors.to_rgb(base_color))
    light_rgb = rgb + (1 - rgb) * max_lightness
    factors = np.linspace(min_intensity, 1, n)
    colors = np.array([light_rgb * f for f in factors])
    return np.flip(colors, axis=0)

def datetime_to_doyfrac(datetimes):
    df = pd.DataFrame()
    df['date'] = datetimes
    df['soy'] = pd.to_datetime(df['date'].dt.year.astype(str) + '-01-01')
    doy = (df['date'] - df['soy']).dt.total_seconds()/86400
    return doy
    