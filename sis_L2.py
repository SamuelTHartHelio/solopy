#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:35:51 2026

@author: shart
"""

import re
import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.colors
import spacepy.pycdf as pycdf
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import matplotlib.patheffects as path_effects
plt.rcParams["font.family"] = "Verdana"
plt.rcParams['ps.fonttype'] = 42 

#----------Local Imports---------#
if os.path.dirname(os.getcwd()) not in sys.path: sys.path.insert(0,os.path.dirname(os.getcwd())) 
from solopy import solopy_utils as su

class SIS_L2():
    
    def __init__(self, start, stop, cadence, output_dir=None):
        self.start, self.stop = start, stop
        self.input_dir = '/Users/shart/Nextcloud/Shared/NC_Hart/solo/Data/'
        self.output_dir = output_dir
        self.cadence = cadence
        self.rate, self.factor = self._sis_rate(cadence)
        self.a_files, self.b_files = self._find_sis(start, stop)
        self.A, self.B, self.C = self._load_sis(start, stop, cadence)
        self._load_spectra()

    def _find_sis(self, start, stop):
        files = np.sort(glob.glob(self.input_dir+'sis_%s/'%self.rate+'*.cdf'))
        latest = {}
        for f in files:
            name = os.path.basename(f)
            match = re.search(r'sis-([ab])-.*_(\d{8})_v(\d+)\.cdf$', name)
            if match is None: continue
            
            s, date, version = match.groups()
            date = pd.to_datetime(date)
            version = int(version)
            if start.date() <= date.date() <= stop.date():
                key = (s, date)
                if key not in latest or version > latest[key][0]:
                    latest[key] = (version, f)
        a_to_use = [v[1] for k, v in sorted(latest.items()) if k[0] == 'a']
        b_to_use = [v[1] for k, v in sorted(latest.items()) if k[0] == 'b']
        return a_to_use, b_to_use

        
    def _sis_rate(self, cadence):
        cadence = pd.to_timedelta(cadence)
    
        rates = {'3sec':  pd.Timedelta('3sec'), 
                 '30sec': pd.Timedelta('30sec'),
                 '30min': pd.Timedelta('30min')}
    
        valid = [r for r, t in rates.items() if t <= cadence]
        if not valid: raise ValueError("Cadence must be >= 3sec.")
    
        rate = valid[-1]
        factor = cadence / rates[rate]
    
        if factor != int(factor):
            raise ValueError(f"Cadence '{cadence}' is not an integer multiple of {rate}.")
        return rate, int(factor)


    def _load_sis(self, start, stop, cadence='3sec'):
        datasets = []
        for files in [self.a_files, self.b_files]:
            cdfs = [pycdf.CDF(f) for f in files]
            sis = pycdf.concatCDF(cdfs)
            tmask = np.where((sis['EPOCH'] >= start) & (sis['EPOCH'] <= stop))[0]
            sis['EPOCH'] = pd.to_datetime(sis['EPOCH'][tmask])
            sis['DELTA_EPOCH'] = sis['DELTA_EPOCH'][tmask]
            cols = list(sis.keys())[2:38]
            for col in cols:
                sis[col] = sis[col][tmask]
            datasets.append(sis)
        sis_a, sis_b = datasets
    
        # --- Rebin --- #
        species = ['H','He3','He4','C','N','O','Ne','Mg','Si','S','Ca','Fe']
        if self.factor > 1:
            for sis in [sis_a, sis_b]:
                nbins = sis['EPOCH'].size // self.factor
                trim = nbins * self.factor
                sis['EPOCH'] = (sis['EPOCH'][:trim:self.factor]+ pd.Timedelta(self.cadence) / 2)
                sis['DELTA_EPOCH'] = (sis['DELTA_EPOCH'][:trim:self.factor] * self.factor)
                for sp in species:
                    for name in ['Flux', 'Rate', 'Uncertainty']:
                        key = sp+'_'+name
                        data = sis[key][:trim]
                        data[data < 0] = 0
                        data = data.reshape(nbins, self.factor, sis[sp+'_Bins_Low_Energy'].size)
                        if name == 'Uncertainty': 
                            sis[key] = np.sqrt((data**2).sum(axis=1)) / self.factor
                        else:
                            sis[key] = data.mean(axis=1)
    
        # --- Combine SIS-A + SIS-B --- #
        sis_c = sis_a.copy()
        for sp in species:
            for name in ['Flux', 'Rate']:
                key = sp+'_'+name
                sis_c[key] = np.nansum([sis_a[key],sis_b[key]],axis=0)
    
            key = sp+'_Uncertainty'
            sis_c[key] = np.sqrt(sis_a[key].astype(float)**2 + sis_b[key].astype(float)**2)
        return sis_a, sis_b, sis_c
    

    def _load_spectra(self):
        for dat in [self.A, self.B, self.C]:
            species = ['H','He3','He4','C','N','O','Ne','Mg','Si','S','Ca','Fe']
            dt = pd.to_datetime(dat['EPOCH'])
        
            for sp in species:
                energy = dat[sp+'_Bins_Low_Energy']
                fluences = np.zeros(energy.size)
                fluences_err = np.zeros(energy.size)
                
                fluences = np.sum(dat[sp+'_Flux'],axis=0)*pd.Timedelta(self.cadence).total_seconds()
                fluences_err = np.sqrt(np.sum(dat[sp+'_Uncertainty'].astype(float)**2, axis=0)) * pd.Timedelta(self.cadence).total_seconds()
                dat[sp+'_Fluence'] = fluences
                dat[sp+'_Fluence_Err'] = fluences_err
    
    def relative_abundances(self,scope='C', energy=0.38, rel_species='O'):
        dat = getattr(self, scope)
        species = ['He3','He4','C','N','O','Ne','Mg','Si','S','Ca','Fe']
        
        ratio = np.zeros(len(species))
        ratio_err = np.zeros(len(species))
        for s, sp in enumerate(species):
            energies = dat[sp+'_Bins_Low_Energy']
            e_idx = np.abs(energies-energy).argmin()
            
            rel_energies = dat[rel_species+'_Bins_Low_Energy']
            rel_e_idx = np.abs(rel_energies-energy).argmin()
            ratio[s] = dat[sp+'_Fluence'][e_idx]/dat[rel_species+'_Fluence'][rel_e_idx]
            ratio_err[s] = ratio[s] * np.sqrt((dat[sp+'_Fluence_Err'][e_idx]/dat[sp+'_Fluence'][e_idx])**2 + 
                                        (dat[rel_species+'_Fluence_Err'][rel_e_idx]/dat[rel_species+'_Fluence'][rel_e_idx])**2)
            
        return species, ratio, ratio_err
        
        
    def species_tseries(self, ax, scope='C', species='He4', spec=1, erange=None, colorbar=True, colors=None):
        dat = getattr(self, scope)
        util = su.species_utils(species)
        energy = dat[species+'_Bins_Low_Energy']+dat[species+'_Bins_Width']/2
        flux = dat[species+'_Flux']
        if erange is None:
            erange = util['erange']
        energies = energy[((energy >= erange[0]) & (energy <= erange[1]))]
        flux = flux[:,((energy >= erange[0]) & (energy <= erange[1]))]
        
        if spec == 0:
            if colors is None:
                colors = su.monochrome_colors(util['color'],energies.size)
            for e, en in enumerate(energies):
                p, = ax.plot(dat['EPOCH'], flux[:,e], color=colors[e],lw=1.25)
            ax.set_ylabel('Diff. Intensity [pfu]')
        if spec == 1:
            cmap = plt.cm.turbo
            cmap.set_bad((0,0,0))
            norm = matplotlib.colors.LogNorm(vmin=1e-2,vmax=1e3)
            espec = (flux*energies**2).T
            p = ax.pcolormesh(dat['EPOCH'], energies, espec, cmap=cmap, norm=norm, rasterized=True)
            ax.annotate(util['name'], xy=(0.01,0.96), xycoords='axes fraction', ha='left',
                        va='top', fontsize=16, color='white',fontweight='bold')
            ax.set_ylim(util['erange'])
            ax.set_ylabel('Energy [MeV/n]',fontsize=9)
            if colorbar:
                cbar = plt.colorbar(p,cax=ax.inset_axes([1.01,0.01,0.017,0.99]))
                cbar.set_label('Diff. Intensity [pfu] * E$^2$', 
                               fontsize=9)
                cbar.ax.tick_params(labelsize=9)
                
        ax.set_yscale('log')
        return p
    
    def species_tseries_AB(self, ax, species='He4', erange=None):
        pA = self.species_tseries(ax, scope='A', species=species, spec=0, erange=erange, colors='blue')
        pB = self.species_tseries(ax, scope='B', species=species, spec=0, erange=erange, colors='red')
        ax.legend([pA, pB], ['SIS-A','SIS-B'])
        
    def species_tseries_AB_ratio(self, ax, species='He4', erange=None, **kwargs):
        datA = self.A
        datB = self.B
        util = su.species_utils(species)
        energy = (datA[species+'_Bins_Low_Energy'] + datA[species+'_Bins_Width']/2)
        if erange is None:
            erange = util['erange']
        idx = (energy >= erange[0]) & (energy <= erange[1])
        energies = energy[idx]
        xx = np.where(idx == True)[0][0]
        print (xx)
        print (datA[species+'_Bins_Low_Energy'][xx:xx+2])
        fluxA = datA[species+'_Flux'][:, idx]
        fluxB = datB[species+'_Flux'][:, idx]
        ratio = fluxA / fluxB
    
        for e, en in enumerate(energies):
            ax.plot(datA['EPOCH'], ratio[:, e], **kwargs)
        ax.axhline(1, color='gray', ls='--', lw=1)
        ax.set_xlim(self.start, self.stop)
        ax.set_yscale('log')
        ax.set_ylabel('SIS-A/SIS-B')
        return ratio
    
    def summary_tseries(self, scope='C', spec=1):
        species = ['He3','He4','C','O','Fe']
        
        fig = plt.figure(figsize=(8,8)) 
        gs = fig.add_gridspec(5,1)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])
        ax2 = fig.add_subplot(gs[2,0])
        ax3 = fig.add_subplot(gs[3,0])
        ax4 = fig.add_subplot(gs[4,0])
        axs = [ax0,ax1,ax2,ax3,ax4]
        for s, sp in enumerate(species):
            p = self.species_tseries(axs[s], scope=scope, species=sp, spec=spec, colorbar=False)
        su.curate_axes(axs,self.start,self.stop,year=self.start.year)
        cbar = plt.colorbar(p,cax=ax4.inset_axes([1.01,0.01,0.017,5.15]))
        cbar.set_label('Differential Intensity [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ (MeV/n)$^{-1}$)] * E$^2$', 
                       fontsize=9)
        cbar.ax.tick_params(labelsize=9)
        fig.subplots_adjust(hspace=0.05,top=0.96,bottom=0.08,left=0.08)
        return fig, axs
    
    def _curate_spectra(self, scope='C', species='He4', frac=1, erange=[0,10]):
        dat = getattr(self, scope)
        energy = dat[species+'_Bins_Low_Energy']+dat[species+'_Bins_Width']/2
        fluence, fluence_err = dat[species+'_Fluence'],dat[species+'_Fluence_Err']
        energy,fluence, fluence_err = np.asarray(energy), np.asarray(fluence), np.asarray(fluence_err)
        if species == 'C' or species == 'O':
            idx = np.abs(energy-0.13).argmin()
            energy = np.delete(energy, idx)
            fluence = np.delete(fluence, idx)
            fluence_err = np.delete(fluence_err, idx)
        fluence[fluence <= 0] = np.nan
        fluence[fluence_err/fluence > frac] = np.nan
        e_idx = ((energy >= erange[0]) & (energy <= erange[1]))
        return energy[e_idx], fluence[e_idx], fluence_err[e_idx] 

    
    def plot_spectra(self, ax, scope='C', species='He4', frac=1, erange=[0,10], fit=None, **kwargs):
        util = su.species_utils(species)
        energy, fluence, fluence_err = self._curate_spectra(scope=scope, species=species, frac=frac, erange=erange)
        
        p = ax.errorbar(energy, fluence, fluence_err, mfc='orange', color='orange', **kwargs) #mew=0.75, color = util['color']
        
        # Fit spectrum
        if fit is not None:
            if fit == 'powerlaw': func = self.power_law
            if fit == 'band': func = self.band_function
            if fit == 'inv_rollover': func = self.inverse_rollover
            
            fit, fit_u = self.fit_spectra(scope=scope, species=species, frac=frac, erange=erange, fit=fit)
            
            efit = np.logspace(np.log10(0.9*energy.min()),np.log10(1.1*energy.max()),500)
            ffit = func(efit,*fit)
            fit = ax.plot(efit, ffit, color=util['color'], linestyle='-.',lw=1.25)
        ax.set_yscale('log')
        ax.set_xscale('log')
        return p
    
    def summary_spectra(self, scope='C', frac=1, erange=[0, 10], xlim=None, ylim=None, **kwargs):
        species = ['H','He3','He4','O','Fe']
        fig = plt.figure(figsize=(4.5,4.5))
        ax = fig.add_subplot()
        for sp in species:
            sis.plot_spectra(ax, species=sp, scope=scope, frac=frac, erange=erange, **kwargs)

        ax.set_ylabel('Fluence [cm$^{-2}$ sr$^{-1}$ (MeV/nuc)$^{-1}$ ]',fontsize=10)
        ax.set_xlabel('Energy [MeV/nuc.]',fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        su.tick_parameters(ax, labelbottom=True)
        fig.tight_layout()

if __name__ == '__main__':
    #start, stop = pd.to_datetime('2024-04-08T16:00:00'), pd.to_datetime('2024-04-10T00:59:00')
    start, stop = pd.to_datetime('2025-05-16T02:00:00'), pd.to_datetime('2025-05-16T16:00:00')

    output_dir = '/Users/shart/Nextcloud/Shared/NC_Hart/solo/Async_SEPs/Figures/'
    sis = SIS_L2(start, stop, cadence='30min', output_dir=output_dir)
    #sis.summary_spectra(scope='C',ylim=[1e0,1e7],marker='o')
    #sis.relative_abundances(scope='C', energy=0.23, rel_species='He3')
    
    fig = plt.figure(figsize=(9,1.5))
    ax = fig.add_subplot()
    sis.species_tseries_AB_ratio(ax, species='He4', erange=[0.7,0.8],color='blue',marker='o',ms=4,lw=1)
    sis.species_tseries_AB_ratio(ax, species='He3', erange=[0.7,0.8],color='red',marker='o',ms=4,lw=1)
    fig.savefig(output_dir+'AB_ratio.eps',dpi=600)
    
























    def power_law(self, E, A, gamma):
        return A * E**(-gamma)
    
    def band_function(self, E, A, alpha, beta, E0):
        E = np.asarray(E)
        Eb = (beta-alpha) * E0

        spectrum = np.empty_like(E, dtype=float)
        low = E <= Eb
        high = ~low

        spectrum[low] = (A * (E[low])**-alpha * np.exp(-E[low] / E0))
        spectrum[high] = (A * (E[high])**-beta) * ((beta - alpha) * E0)**(beta - alpha) * np.exp(alpha - beta) 
        return spectrum
    
    def inverse_rollover(self, E, A, beta, E0):
        return A * E **(-beta)*(np.exp(-E0/E))
    
    def rollover(self, E, A, beta, E0):
        return A * E **(-beta)*(np.exp(-E/E0))
        
    def fit_spectra(self, scope='C', species='He4', frac=1, erange=[0,10], fit=None):
        energy, fluence, fluence_err = self._curate_spectra(scope=scope, species=species, frac=frac, erange=erange)
        idx = np.isfinite(energy) & np.isfinite(fluence) & np.isfinite(fluence_err)
        energy, fluence, fluence_err = energy[idx], fluence[idx], fluence_err[idx]
        
        if fit == 'powerlaw': 
            func = self.power_law
            p0 = [np.nanmax(fluence), 2]
        elif fit == 'band': 
            func = self.band_function
            p0 = [np.nanmax(fluence), 1, 5, 0.2]
        elif fit == 'inv_rollover':
            func = self.inverse_rollover
            p0 = [np.nanmax(fluence), 5, 0.2]
        
        popt, pcov = curve_fit(func, energy, fluence, sigma=fluence_err, absolute_sigma=True, p0=p0)
        print ('---- '+species+' ----')
        print (*popt)
        print (*np.sqrt(np.diag(pcov)))
        return popt, np.sqrt(np.diag(pcov))
    
    
    
