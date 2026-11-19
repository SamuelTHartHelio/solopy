#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:38:20 2026

@author: shart
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams['ps.fonttype'] = 42 

#----------Local Imports---------#
if os.path.dirname(os.getcwd()) not in sys.path: sys.path.insert(0,os.path.dirname(os.getcwd())) 
from solopy import read_solo
from solopy import solo_spectra
from solopy import solopy_utils as su

DATA_DIR = '/Users/shart/Documents/Work/SOSIS/Data/PHA/txt/'

class SIS_PHA():
    
    def __init__(self, start, stop, data_dir=None, output_dir=None, spectra_file=None):
        self.start, self.stop = start, stop
        self.input_dir = DATA_DIR
        self.output_dir = output_dir
        self.files = self._find_pha(self.start, self.stop)
        self.dat = self.load_pha(self.start, self.stop, self.files)
        if spectra_file: 
            self.spectra = self.load_pha_spectra(spectra_file)
        
    def _find_pha(self, start, stop):
        files = np.sort(glob.glob(self.input_dir+'*'))
        dates = pd.date_range(start, stop, freq='1h')
        date_strings = ['_%d_%02d_%02d'%(d.year, d.month, d.day) for d in dates]
        matched = [f for f in files if any(ds in f for ds in date_strings)]
        A_to_use = [f for f in matched if '_A_' in f]
        B_to_use = [f for f in matched if '_B_' in f]
        if not A_to_use: raise FileNotFoundError("No PHA files found for telescope A.")
        if not B_to_use: raise FileNotFoundError("No PHA files found for telescope B.")
        return A_to_use, B_to_use
    
    def _trim_pha(self, start, stop, pha):
        scopes = ['A','B','C']
        for s, scope in enumerate(scopes):
            tmask = np.where((pha[scope]['date'] >= start) & (pha[scope]['date'] <= stop))[0]
            pha[scope] = pha[scope].loc[tmask].reset_index(drop=True)
        return pha

    def load_pha(self, start, stop, all_files):
        scopes = ['A','B']
        pha = {} 
        for scope, files in zip(scopes, all_files):
            dfs = []
            for j, f in enumerate(files):
                df = pd.read_csv(f,skiprows=5)                
                df = df.drop('Unnamed: 23',axis=1)
                df[df[scope+' mass tof3 (amu)'] > 0.2]
                dates = np.array(df['date'])
                
                dates = df['date'].str.split(expand=True)
                dates[0] = dates[0].str.zfill(10)
                dates[1] = dates[1].str.zfill(8)
                df['date'] = pd.to_datetime(dates[0] + 'T' + dates[1])
    
                df[' doy'] = su.datetime_to_doyfrac(df['date'])            
                df['scope'] = scope
                df = df.reset_index(drop=True)
                dfs.append(df)
            pha[scope] = pd.concat(dfs, ignore_index=True)
        
        pha['C'] = pd.concat([pha['A'],pha['B']],ignore_index=True)
        cols = pha['C'].columns
        drop = []
        for col in cols:
            if col.startswith('A '):
                suffix = col[2:]
                bcol = 'B ' + suffix
                if bcol in cols:
                    ccol = 'C ' + suffix
                    pha['C'][ccol] = (pha['C'][col].fillna(pha['C'][bcol]))
                    drop.extend([col, bcol])
        pha['C'] = pha['C'].drop(columns=drop)
        pha['C'] = pha['C'].sort_values('date').reset_index(drop=True)
        trimmed_pha = self._trim_pha(start, stop, pha)
        return trimmed_pha 
    
    def load_pha_spectra(self,file):
        # The first 201 lines are the SIS header.
        dat = pd.read_csv(file, skiprows=201)
        species = ['H', 'He3', 'He4', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']
        energy = dat['MeV/nucleon'].to_numpy()
        energy_width = dat['Delta E'].to_numpy()

        new_spectrum = np.ones(len(energy), dtype=bool)
        new_spectrum[1:] = energy[1:] < energy[:-1]

        # Indices defining the individual spectra
        starts = np.where(new_spectrum)[0]
        stops = np.append(starts[1:], len(energy))
        spectra = {}
        for i, sp in enumerate(species):
            spectra[sp+'_energy'] = energy[starts[i]:stops[i]]
            flux_index = 7 + 2 * i
            err_index = flux_index + 1
            spectra[sp] = dat.iloc[starts[i]:stops[i], flux_index].to_numpy()
            spectra[sp + '_err'] = dat.iloc[starts[i]:stops[i], err_index].to_numpy()
        return spectra

    #-------Load Mass and Inverse Ion Speed Spectrogram--------#
    def load_hehist(self, scope='A', tof=2, mbins=160, mrange=[2, 6], erange=[0.4, 10]):
        tof = np.atleast_1d(tof) 
        df = self.dat[scope]
        epn = df[scope + ' einc tof%d (MeV/nuc)' % tof[-1]]
    
        e_idx = np.where((epn > erange[0]) & (epn < erange[1]))[0]
        masses = []
        for t in tof:
            mass = df[scope + ' mass tof%d (amu)' % t]
            masses.append(mass.iloc[e_idx].values)
        masses = np.array(masses)
    
        mass_min = np.nanmin(masses, axis=0)
        mass_max = np.nanmax(masses, axis=0)
    
        agree = (mass_max / mass_min) <= 1.10
        valid = np.all(np.isfinite(masses), axis=0)
        e_mass = masses[-1, agree & valid]
        if mrange[1] < 10:
            mass_bins = np.linspace(mrange[0], mrange[1], mbins)
        else:
            mass_bins = np.logspace(np.log10(mrange[0]),np.log10(mrange[1]),mbins)
        mass_spec = np.histogram(e_mass, bins=mass_bins)[0]
        return mass_spec, mass_bins
        
    def load_mass_spec(self, scope='A',tof=2,freq='10min',mbins=160,erange=[0.4,10]):
        df = self.dat[scope]
        date = df['date']
        mass = df[scope+' mass tof%d (amu)'%tof]
        epn = df[scope+' einc tof%d (MeV/nuc)'%tof]
        
        dt_bins = pd.date_range(date.iloc[0],date.iloc[-1],freq=freq)
        mass_bins = np.logspace(0.3,1.9,mbins)
        
        e_idx = np.where((epn > erange[0]) & (epn < erange[1]))[0]
        e_dt, e_mass = date.iloc[e_idx], mass.iloc[e_idx]
        mass_spec = np.histogram2d(e_dt,e_mass,bins=[dt_bins,mass_bins])[0].T
        return mass_spec, dt_bins, mass_bins
    
    def load_oos_spec(self,scope='A',tof=2,freq='10min',ebins=160,mrange=[10,80],exclude_he4=False, exclude_o=False):
        df = self.dat[scope]
        date = df['date']
        mass = df[scope+' mass tof%d (amu)'%tof]
        epn = df[scope+' einc tof%d (MeV/nuc)'%tof]
        
        dt_bins = pd.date_range(date.iloc[0],date.iloc[-1],freq=freq)
        oos_bins = np.linspace(0,20,ebins)
        
        mmin, mmax = mrange
        mass_idx = np.where((mass > mmin) & (mass < mmax))[0]
        # Exclude 4He
        if exclude_he4:
            if mmin < 5 and mmax > 3.5:
                mass_idx = mass_idx[(mass[mass_idx] < 3.5) | (mass[mass_idx] > 5)]
        # Exclude O
        if exclude_o:
            if mmin < 19 and mmax > 15:
                mass_idx = mass_idx[(mass[mass_idx] < 15) | (mass[mass_idx] > 19)]
                
        m_dt, m_epn = date.iloc[mass_idx], epn.iloc[mass_idx]
        oos = 1./(3.0e8*np.sqrt(2.*m_epn/940.)) * (1.495e11/3600.)
        oos_spec = np.histogram2d(m_dt,oos,bins=[dt_bins,oos_bins])[0].T
        return oos_spec, dt_bins, oos_bins
            
    def plot_hehist(self,ax,scope='A',tof=2,mbins=160,mrange=[2,6],erange=[0.4,10]):
        mass_spec, mass_bins = self.load_hehist(scope=scope,tof=tof,mbins=mbins,mrange=mrange,erange=erange)
        
        ax.bar(mass_bins[:-1],mass_spec,width=np.diff(mass_bins),color='black')
        ax.set_ylabel('Counts-per-Bin')
        ax.set_xlabel('Mass (AMU) [%.2f - %.2f MeV/n]'%(erange[0],erange[1]),fontsize=10)
        ax.set_ylim([1,np.max(mass_spec)+np.max(mass_spec)/10])
        ax.set_xlim(mrange)
        return mass_spec, mass_bins

    def plot_mass_spec(self,ax,scope='A',tof=2,freq='10min',mbins=160,erange=[0.4,10],he=0):
        mass_spec, mass_dt, mass = self.load_mass_spec(scope=scope,tof=tof,freq=freq,mbins=mbins,erange=erange)
        
        jetty = plt.cm.jet
        jetty.set_under((1,1,1))
        vmin, vmax = 1, 100#np.max([np.max(mass_spec),20])
        m2d = ax.pcolormesh(mass_dt,mass,mass_spec,cmap=jetty,rasterized=True,\
                            norm=mcolors.LogNorm(vmin=vmin,vmax=vmax))
        ax.set_ylabel('Mass (AMU)\n[%.2f - %.2f MeV/n]'%(erange[0],erange[1]),fontsize=8,labelpad=16)
        ax.set_yscale('log')
        ax.set_xlim(mass_dt[0]-pd.to_timedelta(10,'min'),mass_dt[-1]+pd.to_timedelta(10,'min'))
        if he == 1:
            ax.set_ylim(2,6)
            species = ['$^3$He','$^4$He']
            loc = [0.33,0.57]
            for i, s in enumerate(species):
                ax.annotate(s,xy=(-0.005,loc[i]),xycoords='axes fraction',ha='right',fontsize=9)
        else:
            ax.set_ylim(2,80)
            species = ['$^3$He','$^4$He','C','O','Mg','Si','Fe']
            loc = [0.08,0.15,0.43,0.51,0.61,0.66,0.82]
            for i, s in enumerate(species):
                ax.annotate(s,xy=(1.005,loc[i]),xycoords='axes fraction',ha='left',fontsize=9)
        ax.annotate('SIS-%s'%scope,xy=(0.005,0.97),xycoords='axes fraction',
                       va='top',ha='left',fontsize=16)
        cbar = plt.colorbar(m2d,cax=ax.inset_axes([1.01,0.05,0.015,0.9]))
        cbar.set_label('Counts',fontsize=8)
        cbar.ax.tick_params(labelsize=8)


    def plot_oos_spec(self,ax,scope='A',tof=2,freq='10min',ebins=160,mrange=[10,80],exclude_he4=False, exclude_o=False):
        oos_spec, oos_dt, oos = self.load_oos_spec(scope=scope,tof=tof,freq=freq,ebins=ebins,
                                                   mrange=mrange,exclude_he4=exclude_he4,
                                                   exclude_o=exclude_o)
        
        jetty = plt.cm.jet
        jetty.set_under((1,1,1))
        vmin, vmax = 1, 100#np.max([np.max(oos_spec),20])
        e2d = ax.pcolormesh(oos_dt,oos,oos_spec,cmap=jetty,rasterized=True,
                       norm=mcolors.LogNorm(vmin=vmin,vmax=vmax))
        
        ax.set_ylabel('SIS-%s 1/v (hr/AU)\n[%.1f - %.1f AMU]'%(scope,mrange[0],mrange[1]),fontsize=8)
        if exclude_he4: ax.set_ylabel('SIS-%s 1/v (Hour/AU)\n[%.1f - %.1f AMU]\nExcluding $^4$He'%(scope,mrange[0],mrange[1]),fontsize=8)
        if mrange[1] < 4:
            ax.set_ylim(0.1,8)
            ax.set_yticks([2,4,6,8],['2','4','6','8'])
        elif 4 <= mrange[1] < 10:
            ax.set_ylim(0.1,10)
            ax.set_yticks([2,4,6,8,10],['2','4','6','8','10'])
        else:
            ax.set_ylim(0.1,18)
            ax.set_yticks([4,8,12,16],['4','8','12','16'])  
        ax.annotate('SIS-%s'%scope,xy=(0.005,0.02),xycoords='axes fraction',
                    va='bottom',ha='left',fontsize=16)
        ax.set_xlim(self.start, self.stop)
        cbar = plt.colorbar(e2d,cax=ax.inset_axes([1.01,0.05,0.015,0.9]))
        cbar.set_label('Counts',fontsize=8)
        cbar.ax.tick_params(labelsize=8)
        
        secax = ax.secondary_yaxis("right", functions=(su.oos_to_epn, su.epn_to_oos))
        secax.set_ylim(0.02,10)
        secax.set_ticks([0.023,0.05,0.1,0.5,1,10],[' ',' ',' ',' ',' ',' '],fontsize=6,color='red')
        secax.tick_params(axis='y', colors='red', labelsize=6)
        #secax.set_ylabel('Energy [MeV/nuc.]',fontsize=6,color='red')
    
    def plot_pha(self,scope='A',tof=2,freq='10min',mbins=160,ebins=160,mrange=[10,80],erange=[0.4,10],he=0):
        dates = self.dat[scope]['date']
        start, stop = dates.iloc[0], dates.iloc[-1]
        
        fig = plt.figure(figsize=(7.5,5-he))
        if he == 1:
            gs = fig.add_gridspec(3,1)
            ax = fig.add_subplot(gs[0,0])
            ax.set_ylim(0.1,8)
            ax.set_yticks([2,4,6,8],['2','4','6','8'])
        else:
            gs = fig.add_gridspec(4,1)
            ax = fig.add_subplot(gs[0:2,0])
            ax.set_ylim(0.1,18)
            ax.set_yticks([4,8,12,16],['4','8','12','16'])    
        ax1 = fig.add_subplot(gs[-2:,0],sharex=ax)
    
        self.plot_mass_spec(ax,scope=scope,tof=tof,freq=freq,mbins=mbins,erange=erange,he=he)
        self.plot_oos_spec(ax1,scope=scope,tof=tof,freq=freq,ebins=ebins,mrange=mrange)
        
        axs = [ax,ax1]
        su.curate_axes(axs,start=start,stop=stop)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        #fig.savefig('/Users/shart/Desktop/for_mihir.png',dpi=600)
        return fig, gs, ax
    
    def plot_oos_AB(self, tof=2, freq='10min', ebins=160, mrange=[10,80], exclude_he4=False, dist=None, time=None, vsw=400, axs=None):
        standalone = axs is None
        if standalone:
            fig, axs = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, sharey=True)
        else:
            fig = axs[0].figure
    
        for ax, scope in zip(axs, ['A', 'B']):
            self.plot_oos_spec(ax, scope=scope, tof=tof, freq=freq, ebins=ebins, mrange=mrange, exclude_he4=exclude_he4)
    
        if time is None: time = self.start
        if dist is not None:
            for ax in axs:
                su.dispersion_plot(ax, dist=dist, time=time, vsw=vsw, erange=[0.02, 1e4], space='inverse_v', color='black', ls='-', lw=1.5)
        su.curate_axes(axs, start=self.start, stop=self.stop, year=self.start.year)
        if standalone:
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            fig.savefig(self.output_dir + 'sunward_streaming_AB.png', dpi=600)
        return fig, axs
    
    def fit_pha_spectra(self, energy, flux, flux_err, species=None, fit=None):
        if fit == None:
            raise ValueError('Please choose a function: powerlaw, rollover, band')      
        elif fit == 'powerlaw': 
            func = power_law
            p0 = [np.nanmax(flux), 2]
            bounds = ([-np.inf, 0], [np.inf, np.inf])
        elif fit == 'band': 
            func = band_function
            p0 = [np.nanmax(flux), 1, 5, 0.2]
            bounds = ([-np.inf, 0, -np.inf, -np.inf], [np.inf, np.inf, 10, np.inf])
        elif fit == 'rollover':
            func = rollover
            p0 = [np.nanmax(flux), 2, 0.2]
            bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
        
        popt, pcov = curve_fit(func, energy, flux, sigma=flux_err, absolute_sigma=True, p0=p0, bounds=bounds)
        residuals = flux - func(energy, *popt)
        chisq = np.sum((residuals / flux_err) ** 2)
        dof = len(flux) - len(popt)
        reduced_chisq = chisq / dof
        
        # Rescale errors
        flux_err_new = flux_err * np.sqrt(reduced_chisq)
        popt, pcov = curve_fit(func,energy, flux, sigma=flux_err_new, p0=popt, absolute_sigma=True, bounds=bounds)
        if species: print ('---- '+species+' ----')
        if not species: print ('---- Fit ----')
        print (*popt)
        print (*np.sqrt(np.diag(pcov)))
        print ('Reduced Chi-Squared: %.2f'%reduced_chisq)
        
        return func, popt, np.sqrt(np.diag(pcov))
    
    def plot_pha_spectra(self, ax, species='H', erange=None, fit=None, async_seps=False, **kwargs):
        utils = su.species_utils(species)
        energy = self.spectra[species + '_energy']
        flux = self.spectra[species]
        error = self.spectra[species + '_err']

        good = np.isfinite(energy) & np.isfinite(flux) & np.isfinite(error)
        good &= (flux > 0) & (error >= 0)
        if erange is None: erange = utils['erange']
        good &= (energy >= erange[0]) & (energy <= erange[1])
        eplot, fplot, fplot_err = energy[good], flux[good], error[good]
        if async_seps == True:
            fplot *= 2
            fplot_err *= 2
        ax.errorbar(eplot, fplot, yerr=fplot_err, mew=0.75, mec='black',
                    color=utils['color'], label=utils['name'], lw=0.75, **kwargs)
        if fit is not None:
            func, pars, pars_err = self.fit_pha_spectra(eplot, fplot, fplot_err, species=species, fit=fit)
            efit = np.logspace(np.log10(0.9*eplot.min()),np.log10(1.1*eplot.max()),500)
            ffit = func(efit,*pars)
            l1 = ax.plot(efit, ffit, color=utils['color'], linestyle='-.',lw=1.25)
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy [MeV/nuc.]')
        ax.set_ylabel(r'Fluence [cm$^{-2}$ sr$^{-1}$ (MeV/nuc)$^{-1}$]')
        
        su.tick_parameters(ax,labelbottom=True)
        return ax
    
def power_law(E, A, gamma):
    return A * E**(-gamma)

def band_function(E, A, alpha, beta, E0):
    E = np.asarray(E)
    Eb = (beta-alpha) * E0

    spectrum = np.empty_like(E, dtype=float)
    low = E <= Eb
    high = ~low

    spectrum[low] = (A * (E[low])**-alpha * np.exp(-E[low] / E0))
    spectrum[high] = (A * (E[high])**-beta) * ((beta - alpha) * E0)**(beta - alpha) * np.exp(alpha - beta) 
    return spectrum

def inverse_rollover(E, A, beta, E0):
    return A * E **(-beta)*(np.exp(-E0/E))

def rollover(E, A, beta, E0):
    return A * E **(-beta)*(np.exp(-E/E0))

if __name__ == '__main__':
    output_dir = '/Users/shart/Library/CloudStorage/OneDrive-SWRI/Documents/Presentations/2026/AGU_SolO_IMAP/SolO/'
    start, stop = pd.to_datetime('2024-04-08T14:00:00'), pd.to_datetime('2024-04-10T08:00:00')
    sis_pha = SIS_PHA(start, stop, output_dir=output_dir)
    sis_pha.plot_oos_AB(tof=3,freq='8min',ebins=200,mrange=[10,80],exclude_he4=True)
    #sis_pha.plot_oos_AB(tof=3,freq='12min',ebins=160,mrange=[2,3.5])

    
    