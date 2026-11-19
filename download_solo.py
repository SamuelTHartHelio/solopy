#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 08:08:28 2025

@author: shart
"""

import os
import glob
import pickle
import cdflib
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date, timedelta

#----User Inputs----#
dpath = '/Users/shart/Nextcloud/Shared/NC_Hart/solo/Data/'
update = 1

#----Preparation Code--------#
launch_year = 2020
today_year = date.today().year
path = 'https://cdaweb.gsfc.nasa.gov/pub/data/solar-orbiter/'
if update == 0: years = np.arange(launch_year,today_year+1)
if update == 1: years = np.arange(today_year-1,today_year+1)

#-----Download Functions-------#
def dl_solo_mag(resolution,update=0):
    if resolution == '4sec': ext = ''
    if resolution == '1min': ext = '-1-minute'
    dl_folder = dpath+'mag_%s/'%resolution
    dl_files = [os.path.basename(f) for f in glob.glob(dl_folder+'*')]
    mag_path = path + 'mag/science/l2/rtn-normal%s/'%ext
    for i, y in enumerate(years):
        p2f = requests.get(mag_path+'%d'%y)
        soup = BeautifulSoup(p2f.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            if link['href'].endswith('.cdf'):
                if link['href'] not in dl_files:
                    f2d = requests.get(mag_path+'%d/'%y+link['href'])
                    with open(dl_folder+link['href'], 'wb') as f:
                        f.write(f2d.content)
                    print ('Saving %s...'%link['href'])
    return
#dl_solo_mag(resolution='1min',update=update)

def dl_solo_sis(resolution='all',update=0):
    if resolution == 'all': res = ['slow','medium','fast']
    if resolution != 'all': res = [resolution]
    for r in res:
        if r == 'slow': dl_folder = dpath+'sis_30min/'
        if r == 'medium': dl_folder = dpath+'sis_30sec/'
        if r == 'fast': dl_folder = dpath+'sis_3sec/'
        dl_files = [os.path.basename(f) for f in glob.glob(dl_folder+'*')]
        scope = ['a','b']
        for j, s in enumerate(scope):
            sis_path = path+'epd/science/l2/sis/'+s+'-rates-'+r+'/'
            for i, y in enumerate(years):
                p2f = requests.get(sis_path+'%d'%y)
                soup = BeautifulSoup(p2f.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    if link['href'].endswith('.cdf'):
                        if link['href'] not in dl_files:
                            f2d = requests.get(sis_path+'%d/'%y+link['href'])
                            with open(dl_folder+link['href'], 'wb') as f:
                                f.write(f2d.content)
                            print ('Saving %s...'%link['href'])
#dl_solo_sis(resolution='all',update=update)

def dl_solo_pas_moments(update=0):
    dl_folder = dpath+'pas_moments/'
    dl_files = [os.path.basename(f) for f in glob.glob(dl_folder+'*')]
    pas_path = path + 'swa/science/l2/pas-grnd-mom/'
    for i, y in enumerate(years):
        p2f = requests.get(pas_path+'%d'%y)
        soup = BeautifulSoup(p2f.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            if link['href'].endswith('.cdf'):
                if link['href'] not in dl_files:
                    f2d = requests.get(pas_path+'%d/'%y+link['href'])
                    with open(dl_folder+link['href'], 'wb') as f:
                        f.write(f2d.content)
                    print ('Saving %s...'%link['href'])
    return
#dl_solo_pas_moments(update)       

def dl_solo_eas(update=0):
    dl_folder = dpath+'eas/'
    dl_files = [os.path.basename(f) for f in glob.glob(dl_folder+'*')]
    eas_path = path + 'swa/science/l3/eas-nmpad-psd/'
    for i, y in enumerate(years):
        p2f = requests.get(eas_path+'%d'%y)
        soup = BeautifulSoup(p2f.text, 'html.parser')
        for l, link in enumerate(soup.find_all('a', href=True)):
            if link['href'].endswith('.cdf'):
                if link['href'] not in dl_files:
                    f2d = requests.get(eas_path+'%d/'%y+link['href'])
                    with open(dl_folder+link['href'], 'wb') as f:
                        f.write(f2d.content)
                    print ('Saving %s...'%link['href'])
    return
dl_solo_eas(update)                

def dl_solo_ept(update=0):
    dl_folder = dpath+'ept_1min/'
    dl_files = [os.path.basename(f) for f in glob.glob(dl_folder+'*')]
    ept_path = path + 'epd/science/l3/ept-1min/'
    for i, y in enumerate(years):
        p2f = requests.get(ept_path+'%d'%y)
        soup = BeautifulSoup(p2f.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            if link['href'].endswith('.cdf'):
                if link['href'] not in dl_files:
                    f2d = requests.get(ept_path+'%d/'%y+link['href'])
                    with open(dl_folder+link['href'], 'wb') as f:
                        f.write(f2d.content)
                    print ('Saving %s...'%link['href'])
#dl_solo_ept(update)       

        
        
        
        
        
        
        
        