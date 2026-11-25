#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 08:17:48 2025

@author: shart
"""

import re
from contextlib import ExitStack
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
import matplotlib.patheffects as path_effects
#plt.rcParams["font.family"] = "Helvetica"

#----------Local Imports---------#
if os.path.dirname(os.getcwd()) not in sys.path: sys.path.insert(0,os.path.dirname(os.getcwd())) 
from solopy import solopy_utils as gf

L2_path = '/Users/shart/Nextcloud/Shared/NC_Hart/solo/Data/'


    

#-------MAG--------#
def find_mag(start, stop, resolution, *, data_dir=None):
    root = L2_path if data_dir is None else data_dir
    all_files = np.sort(glob.glob(os.path.join(root, f'mag_{resolution}', '*')))
    dates = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(stop).normalize())
    suffixes = ['_%d%02d%02d_'%(d.year, d.month, d.day) for d in dates]
    to_use = [f for f in all_files if any(s in f for s in suffixes)]
    return to_use

def load_mag(start, stop, resolution, *, data_dir=None):
    files = find_mag(start, stop, resolution, data_dir=data_dir)
    if not files:
        root = L2_path if data_dir is None else data_dir
        raise FileNotFoundError(f"No MAG files for the selected dates in {os.path.join(root, f'mag_{resolution}')}")
    cdfs = [pycdf.CDF(f) for f in files]
    mag = pycdf.concatCDF(cdfs)
    mag['EPOCH'] = pd.to_datetime(mag['EPOCH'])
    b_tot = np.sqrt(np.sum(mag['B_RTN']**2,axis=1))
    mag['B_RTN'] = np.hstack((mag['B_RTN'],np.array([b_tot]).T))
    mag['LBL1_B_RTN'] = np.array(['B$_R$','B$_T$','B$_N$','|B|'])
    mag['B_th'] = np.arcsin(mag['B_RTN'][:,2]/mag['B_RTN'][:,3])*180/np.pi
    mag['B_ph'] = np.arctan2(mag['B_RTN'][:,0],-mag['B_RTN'][:,1])*180/np.pi
    sisa = np.array([np.cos(150*np.pi/180),np.sin(150*np.pi/180),0])
    sisb = np.array([np.cos(20*np.pi/180),np.sin(20*np.pi/180),0])
    mag['A_pad'] = np.arccos(np.dot(sisa,mag['B_RTN'][:,0:3].T)/np.sqrt(np.sum(mag['B_RTN'][:,0:3]**2,1)))*180/np.pi
    mag['B_pad'] = np.arccos(np.dot(sisb,mag['B_RTN'][:,0:3].T)/np.sqrt(np.sum(mag['B_RTN'][:,0:3]**2,1)))*180/np.pi
    return mag

#--------PAS Moments---------#
def find_pas(start, stop):
    all_files = np.sort(glob.glob(L2_path+'pas_moments/*'))
    dates = pd.date_range(start, stop)
    suffixes = ['_%d%02d%02d_' % (d.year, d.month, d.day) for d in dates]
    to_use = [f for f in all_files if any(s in f for s in suffixes)]
    return to_use

def load_pas(start, stop):
    files = find_pas(start,stop)
    cdfs = [pycdf.CDF(f) for f in files]
    pas = pycdf.concatCDF(cdfs)
    return pas

#-------EPT Public--------#
def find_ept(start, stop, *, data_dir=None):
    root = L2_path if data_dir is None else data_dir
    all_files = sorted(glob.glob(os.path.join(root, 'ept_1min', '*.cdf')))
    dates = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(stop).normalize())
    suffixes = ['_%d%02d%02d_' % (d.year, d.month, d.day) for d in dates]
    to_use = [f for f in all_files if any(s in f for s in suffixes)]
    return to_use

def load_ept(start, stop, *, data_dir=None):
    files = find_ept(start, stop, data_dir=data_dir)
    if not files:
        raise FileNotFoundError("No ept_1min CDFs found for the selected dates")
    with ExitStack() as stack:
        cdfs = [stack.enter_context(pycdf.CDF(f)) for f in files]
        return pycdf.concatCDF(cdfs)


#-------EAS Strahl--------#
def find_eas(start, stop):
    all_files = np.sort(glob.glob(L2_path+'eas/*'))
    dates = pd.date_range(start, stop)
    suffixes = ['_%d%02d%02d_' % (d.year, d.month, d.day) for d in dates]
    to_use = [f for f in all_files if any(s in f for s in suffixes)]
    return to_use

def load_eas(start,stop):
    files = find_eas(start,stop)
    cdfs = [pycdf.CDF(f) for f in files]
    eas = pycdf.concatCDF(cdfs)
    return eas

def load_eas_rebinned(start, stop, pa_factor=1, t_factor=1, ignore_nan=False):
    eas = load_eas(start, stop)
    time = eas['EPOCH']
    pitch = eas['SWA_EAS_PA']
    energy = eas['SWA_EAS_ENERGY']
    pad3d = eas['SWA_EAS_NMPAD_PSD_DATA']
    
    n_t, n_pa, n_e = pad3d.shape
    n_pa_trim = n_pa - n_pa % pa_factor
    n_t_trim  = n_t  - n_t  % t_factor

    pad3d  = pad3d[:n_t_trim,:n_pa_trim,:]
    pitch = pitch[:n_pa_trim]
    time  = time[:n_t_trim]
    data_blocks = pad3d.reshape(n_t_trim  // t_factor, t_factor,
                                n_pa_trim // pa_factor, pa_factor,
                                n_e)

    if ignore_nan: reducer = np.nanmean
    else: reducer = np.mean

    # Reduce pitch and time block axes
    eas['SWA_EAS_NMPAD_PSD_DATA'] = reducer(data_blocks, axis=(1,3))
    eas['SWA_EAS_PA'] = pitch.reshape(-1, pa_factor).mean(axis=1)
    eas['EPOCH'] = time[int(t_factor/2)::t_factor]
    return eas

#-------SW Eflux Public--------#
def find_eflux(start,stop):
    suf_start = '_%d%02d%02d_'%(start.year,start.month,start.day)
    suf_stop = '_%d%02d%02d_'%(stop.year,stop.month,stop.day)
    files = np.sort(glob.glob('/Users/shart/Nextcloud/Shared/NC_Hart/solo/Data/pas_eflux/'+'*'))
    to_use = files[np.where(np.char.find(files,suf_start) != -1)[0][0]:\
                   np.where(np.char.find(files,suf_stop) != -1)[0][-1]+1]
    return to_use

def load_eflux(start, stop):
    files = find_eflux(start,stop)
    cdfs = [pycdf.CDF(f) for f in files]
    eflux = pycdf.concatCDF(cdfs)
    return eflux


#--------------------Finding and Loading Data--------------------#



#-------------------Compton-Getting Corrections------------------#
def downsample_sw_speed(sis_t, sw_t, sw_v):
    #For SO/PSP, use uncorrected solar wind speed in spacecraft frame
    #At L1 or beyond, any V_RTN will suffice.
    u_rel = sw_v.copy().astype(float)
    u_rel[:,:2] *= -1
    
    #----Downsampling 3D SW to SIS Cadence----#
    sis_dt = ((sis_t[1:] - sis_t[:-1]) / 2).mean()
    sw_dt = ((sw_t[1:] - sw_t[0:-1]) / 2).mean()
    sw_t = pd.to_datetime(sw_t)
    
    span = int(sis_dt/sw_dt/2)
    u_resample, u_unc = np.zeros(shape=(sis_t.size,3)), np.zeros(shape=(sis_t.size,3))
    for i in np.arange(sis_t.size):
        idx = np.abs(sw_t-sis_t[i]).argmin()
        u_resample[i,:] = np.mean(u_rel[idx-span:idx+span,:],axis=0)
        u_unc[i,:] = np.std(u_rel[idx-span:idx+span,:],axis=0)
    return u_resample, u_unc
    
def sis_cg_corr(sis_t, energies, flux_a, flux_b, plasma, gamma=-1):
    u_resample, u_unc = downsample_sw_speed(sis_t, plasma['Epoch'], plasma['V_SRF'])

    v_sep = np.sqrt(2*1.602e-13*energies/1.67e-27)/1e3
    sisa_look, sisb_look = np.array([0.866,-0.5,0]), np.array([-0.94,-0.34,0])

    sis_look = [sisa_look, sisb_look]
    sis_flux = [flux_a, flux_b]
    for i in range(2):
        v = v_sep[None, :, None] * sis_look[i][None, None, :]    
        vp = v - u_resample[:, None, :]
        norm_v, norm_vp = np.linalg.norm(v, axis=2), np.linalg.norm(vp, axis=2)
        cg_fct = (norm_vp/norm_v)**2
        cg_e = 0.5*1.67e-27*(1e3*norm_vp)**2/1.602e-13
        cg_flux = sis_flux[i]*cg_fct
    
        # Uncertainty propagation by perturbation
        vp_plus  = v - (u_resample + u_unc)[:, None, :]
        vp_minus = v - (u_resample - u_unc)[:, None, :]
    
        norm_vp_plus  = np.linalg.norm(vp_plus,  axis=2)
        norm_vp_minus = np.linalg.norm(vp_minus, axis=2)

        cg_fct_plus  = (norm_vp_plus/norm_v)**2
        cg_fct_minus = (norm_vp_minus/norm_v)**2
    
        cg_fct_unc = np.abs(cg_fct_plus - cg_fct_minus)
        cg_flux_unc = cg_flux * cg_fct_unc
        
        if gamma != -1:
            #Correcting for energy adjustment
            cg_flux = cg_flux*(cg_e/energies)**(gamma)
            cg_flux_unc = cg_flux_unc*(cg_e/energies)**(gamma) 
        if i == 0: cg_e_a, cg_flux_a, cg_flux_a_unc = cg_e, cg_flux, cg_flux_unc
        if i == 1: cg_e_b, cg_flux_b, cg_flux_b_unc = cg_e, cg_flux, cg_flux_unc
    return cg_e_a, cg_flux_a, cg_flux_a_unc, cg_e_b, cg_flux_b, cg_flux_b_unc

def sis_cg_corr_resampled(energies, flux_a, flux_b, u_resample, u_unc, gamma=-1):
    v_sep = np.sqrt(2*1.602e-13*energies/1.67e-27)/1e3
    sisa_look, sisb_look = np.array([0.866,-0.5,0]), np.array([-0.94,-0.34,0])

    sis_look = [sisa_look, sisb_look]
    sis_flux = [flux_a, flux_b]
    for i in range(2):
        v = v_sep[None, :, None] * sis_look[i][None, None, :]    
        vp = v - u_resample[:, None, :]
        norm_v, norm_vp = np.linalg.norm(v, axis=2), np.linalg.norm(vp, axis=2)
        cg_fct = (norm_vp/norm_v)**2
        cg_e = 0.5*1.67e-27*(1e3*norm_vp)**2/1.602e-13
        cg_flux = sis_flux[i]*cg_fct
    
        # Uncertainty propagation by perturbation
        vp_plus  = v - (u_resample + u_unc)[:, None, :]
        vp_minus = v - (u_resample - u_unc)[:, None, :]
    
        norm_vp_plus  = np.linalg.norm(vp_plus,  axis=2)
        norm_vp_minus = np.linalg.norm(vp_minus, axis=2)

        cg_fct_plus  = (norm_vp_plus/norm_v)**2
        cg_fct_minus = (norm_vp_minus/norm_v)**2
    
        cg_fct_unc = np.abs(cg_fct_plus - cg_fct_minus)
        cg_flux_unc = cg_flux * cg_fct_unc
        
        if gamma != -1:
            #Correcting for energy adjustment
            cg_flux = cg_flux*(cg_e/energies)**(gamma)
            cg_flux_unc = cg_flux_unc*(cg_e/energies)**(gamma) 
        if i == 0: cg_e_a, cg_flux_a, cg_flux_a_unc = cg_e, cg_flux, cg_flux_unc
        if i == 1: cg_e_b, cg_flux_b, cg_flux_b_unc = cg_e, cg_flux, cg_flux_unc
    return cg_e_a, cg_flux_a, cg_flux_a_unc, cg_e_b, cg_flux_b, cg_flux_b_unc

def load_sis_cg_corrected(start,stop,rate,factor=1,gamma=-1):
    sis_a, sis_b = load_sis_rebinned(start,stop,rate,factor=factor)
    pas = load_pas(start, stop)
    
    u_resample, u_unc = downsample_sw_speed(sis_a['EPOCH'], pas['Epoch'], pas['V_SRF'])

    species = ['H','He3','He4','C','N','O','Ne','Mg','Si','S','Ca','Fe']
    epoch = sis_a['EPOCH']
    
    for s, sp in enumerate(species):
        energy = sis_a['%s_Bins_Low_Energy'%sp]+sis_a['%s_Bins_Width'%sp]/2
        sis_a['%s_Energy_CG'%sp], sis_a['%s_Flux_CG'%sp], sis_a['%s_Uncertainty_CG'%sp],\
        sis_b['%s_Energy_CG'%sp], sis_b['%s_Flux_CG'%sp], sis_b['%s_Uncertainty_CG'%sp]=\
        sis_cg_corr_resampled(energy, sis_a['%s_Flux'%sp], sis_b['%s_Flux'%sp], \
                              u_resample, u_unc, gamma=gamma)
    return sis_a, sis_b, pas
    

def cg_ratio(sis_t, sw_t, sw_v, energies, gamma, energy_corr=0):
    u_resample, u_unc = downsample_sw_speed(sis_t, sw_t, sw_v)

    energies = np.atleast_1d(energies)
    v_sep = np.sqrt(2*1.602e-13*energies/1.67e-27)/1e3
    sisa_look, sisb_look = np.array([0.866,-0.5,0]), np.array([-0.94,-0.34,0])

    sis_look = [sisa_look, sisb_look]
    for i in range(2):
        v = v_sep[None, :, None] * sis_look[i][None, None, :]    
        vp = v - u_resample[:, None, :]
        norm_v, norm_vp = np.linalg.norm(v, axis=2), np.linalg.norm(vp, axis=2)
        R = norm_vp/norm_v
        cg_fct = R**2
        cg_e = 0.5*1.67e-27*(1e3*norm_vp)**2/1.602e-13
        if i == 0: cg_e_a, cg_fct_a = cg_e, cg_fct
        if i == 1: cg_e_b, cg_fct_b = cg_e, cg_fct
    
    if energy_corr == 0:
        iso_ratio = 1./(cg_fct_a/cg_fct_b)
        iso_ani = (iso_ratio-1)/(iso_ratio+1)
        return iso_ratio, iso_ani
    
    if energy_corr == 1:
        #Correction to Original Energies
        cg_fct_a_p = cg_fct_a*(cg_e_a/energies)**(gamma)
        cg_fct_b_p = cg_fct_b*(cg_e_b/energies)**(gamma)
        
        iso_corr_ratio = (1./(cg_fct_a_p/(cg_fct_b_p))).flatten()
        iso_corr_ani = ((iso_corr_ratio-1)/(iso_corr_ratio+1)).flatten()
        return iso_corr_ratio, iso_corr_ani
#-------------------Compton-Getting Corrections------------------#




#-------------------------Analyzing Data-------------------------#
def dispersive_onsets(start, stop, p1, p2, nrg):
    AU_km, J_MeV, mp = 1.495e8, 1.602e-13, 1.607e-27
    lag10, lag20 = (p1*AU_km)/(np.sqrt(2*J_MeV*nrg/mp)/1e3), (p2*AU_km)/(np.sqrt(2*J_MeV*nrg/mp)/1e3)
    lag1, lag2 = np.array([timedelta(seconds=int(s)) for s in lag10]), np.array([timedelta(seconds=int(s)) for s in lag20])
    onsets1, onsets2 = pd.to_datetime(start + lag1), pd.to_datetime(stop + lag2)
    return onsets1, onsets2

def spec_poly(start, stop, e_or_iv, startpath,stoppath):
    startlag = 20*startpath
    stoplag = 20*stoppath
    poly1 = start
    poly2 = start+timedelta(hours=startlag)
    poly3 = stop+timedelta(hours=stoplag)
    poly4 = stop
    xpoly = np.array([poly1,poly2,poly3,poly4,poly1])
    ypoly = np.array([0,20,20,0,0])
    if e_or_iv == 'iv':
        return xpoly, ypoly
    if e_or_iv == 'e':
        ypoly = ypoly+0.01
        ypoly0 = np.arange(ypoly[0],ypoly[1],0.01)
        ypoly1 = np.array([ypoly0,np.flip(ypoly0)]).flatten()
        ypolye = (940/2)*((1.495e11/3600)/(3e8*ypoly1))**2
        xpoly0 = pd.date_range(xpoly[0],xpoly[1],periods=ypoly0.size)
        xpoly1 = pd.date_range(xpoly[3],xpoly[2],periods=ypoly0.size)
        xpolye = np.array([xpoly0,np.flip(xpoly1)]).flatten()
        return xpolye, ypolye

def plot_ept_electrons(start,stop,savfig=False):
    ept = load_ept(start,stop)

    cmap = plt.cm.turbo
    #cmap.set_bad((0,0,0))
        
    fig = plt.figure(figsize=(6,5))
    gs = fig.add_gridspec(4,1)
    ax = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0],sharex=ax)
    ax2 = fig.add_subplot(gs[2,0],sharex=ax)
    ax3 = fig.add_subplot(gs[3,0],sharex=ax)
    axs = [ax,ax1,ax2,ax3]
    date_form = mdates.DateFormatter('%H:%M')

    scope = ['S','A','N','D']
    names = ['Sun','Anti-Sun','North','South']
    energy = ept['Electron_Energy']
    energy = energy.reshape(int(energy.size/2),2).mean(axis=1)

    vmin, vmax = 3e0, 3e2      
    for i in np.arange(4):
        espec = ept['Electron_Flux_%s'%scope[i]]
        espec = espec.reshape(espec[:,0].size, energy.size, 2).mean(axis=2)
        eflux = (espec * energy**2)
        eflux[eflux <= 0] = vmin
        l1 = axs[i].pcolormesh(ept['EPOCH'],energy,eflux[:-1,:-1].T,shading='auto',cmap=cmap,
                  norm=matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax),rasterized=True)
        txt = axs[i].annotate(names[i],xy=(0.01,0.8),xycoords='axes fraction',ha='left',
                              color='white',fontsize=15,fontweight='bold')
        txt.set_path_effects([path_effects.Stroke(linewidth=1, foreground="black"),path_effects.Normal()])
        axs[i].set_yscale('log')
        axs[i].set_ylabel('Energy (MeV)')
        axs[i].set_xlim(start,stop)
    ax3.xaxis.set_major_formatter(date_form)
    title = start.strftime('%b %d, %Y')
    ax3.set_xlabel('Date (%s)'%title)
        
    cbar = fig.colorbar(l1,cax=ax3.inset_axes([1.02,0.05,0.017,3.9]))
    cbar.set_ticks([10,100])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Diff. Flux*E$^2$ (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV)',fontsize=10)
    
    fig.subplots_adjust(hspace=0,right=0.85,top=0.95,bottom=0.1)
    if savfig == True:
        fig.savefig('/Users/shart/Nextcloud/Shared/NC_Hart/solo/Inward_3He/Figures/electrons_event1.eps')
    return


#-----------SIS Illustrations or Animations----------#

def draw_sis_fov(ax,projection='polar'):
    th_a = np.array([19,41])*np.pi/180
    th_b = np.array([149,171])*np.pi/180
    ths = [th_a, th_b]
    cs = ['lightgray','darkgray']
    if projection == 'polar':
        for t, th in enumerate(ths):
            theta1 = th[0]
            theta2 = th[1]
            theta = np.linspace(theta1,theta2,100)
            ax.fill_between(theta,ax.get_rmin(),1.1*ax.get_rmax(),color=cs[t],alpha=0.5,zorder=0)
        ax.annotate('SIS-A',xy=(30*np.pi/180,1.08),ha='center',va='center',color='gray',xycoords='data',rotation=60,fontsize=16,annotation_clip=False)
        ax.annotate('SIS-B',xy=(160*np.pi/180,1.07),ha='center',va='center',color='dimgray',xycoords='data',rotation=-70,fontsize=16,annotation_clip=False)
    else:
        ax.axvspan(xmin=th_a[0],xmax=th_a[1],ymin=0,ymax=1,color=cs[0],alpha=0.5,zorder=0)
        ax.axvspan(xmin=th_b[0],xmax=th_b[1],ymin=0,ymax=1,color=cs[1],alpha=0.5,zorder=0)
        ax.annotate('SIS-A',xy=(30*np.pi/180,-0.75),ha='center',va='center',color='gray',xycoords='data',rotation=90,fontsize=16)
        ax.annotate('SIS-B',xy=(160*np.pi/180,-0.75),ha='center',va='center',color='dimgray',xycoords='data',rotation=90,fontsize=16)
    ax.annotate("", xytext=(0,-1), xy=(0,1),arrowprops=dict(arrowstyle="->",lw=1.5,mutation_scale=25))
  
if __name__ == '__main__':
    start = pd.to_datetime('2026-02-05T00:00:00')
    stop = pd.to_datetime('2026-02-06T00:00:00')
    ept = load_ept(start, stop)
    
    fig = plt.figure(figsize=(7,3))
    ax = fig.add_subplot()
    energy = 0.05
    e_idx = np.abs(ept['Electron_Energy']-energy).argmin()
    ax.plot(ept['EPOCH'],ept['Electron_Flux_S'][:,e_idx])
    ax.set_yscale('log')
    ax.set_ylim([1e3,1e5])
    ax.set_ylabel(ept['Electron_Energy'][e_idx])
    #iso = cg_ratio(0,0.2,gamma=2,energy_corr=1)
    #plot_pha_AB(start,stop,freq='5T',ybins=240,m_range=[3.5,5])
    #sis_a, sis_b = load_sis(start,stop,'30sec')







