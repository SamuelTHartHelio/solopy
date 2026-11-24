#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:14:27 2026

@author: shart
"""

import Path
import matplotlib.pyplot as plt
from codicepy.cospice import COSPICE

#------------------------------GENERAL SETTINGS-------------------------------#
KERNEL_PATH = Path.home() / 'anaconda3' / 'krnls' / 'solar_sys_pos.txt'
ORBIT_PATH = KERNEL_PATH.parent  # Contains ace_spice/ and wind_spice/ orbit CDFs
# Use a kernel or metakernel containing the selected targets and time interval.
START = '2023-01-10T00:00:00'
STOP = '2023-12-20T00:00:00'     # None for a single time
CADENCE = '1h'

OBSERVER = 'Sun'               # 'Sun' or 'Earth'
FRAME = 'HEE'           # 'ECLIPJ2000', 'IAU_SUN', 'GSE', or 'HEE'
#SATELLITES = ['Earth', 'Mars Barycenter', 'STEREO Ahead', 'Parker Solar Probe']
SATELLITES = ['Earth','Solar Orbiter']
# ACE/Wind need FRAME='GSE', OBSERVER='Earth', and cospice_utils.py.

SAVE_FIG = 0
SAVE_EXT = '.eps'
SAVE_PATH = str(Path.home()/'Library'/'CloudStorage'/'OneDrive-SWRI'/'Documents'/
                'Work'/'Proposals'/'2026'/'NIAC')

#------------------------------OUTPUT SELECTIONS------------------------------#
CARTESIAN = 0
POLAR = 1

FOOTPOINTS = 0
EPHEMERIS = 0
PLANE = 'XY'                 # Cartesian plane: 'XY', 'XZ', or 'YZ'

#--Polar settings--#
RMAX = 1.1           # Radial limit; AU for Sun-centered plots

# One speed for all targets, or a dictionary with a speed for each SATELLITES entry.
SW_SPEED = 400               # km/s; used by both spirals and source-surface footpoints
# SW_SPEED = {'Earth': 400, 'Solar Orbiter': 500,
#             'Parker Solar Probe': 400, 'STEREO Ahead': 400}

#------------------------------PLOT PARAMETERS--------------------------------#
orbit_p = {'spiral': 0,               # 0 or 1; Parker spirals are for Sun-centered plots
           'source': None,            # Source longitude in degrees, or None
           }

footpoint_p = {'rss': 2.5,                # Source-surface radius in solar radii
               }


if __name__ == '__main__':
    if not (CARTESIAN or POLAR or FOOTPOINTS or EPHEMERIS):
        raise ValueError('Enable at least one COSPICE output.')
    if SAVE_FIG:
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    spice = COSPICE(kernel_path=KERNEL_PATH,output_dir=SAVE_PATH if SAVE_FIG else None,
                    obs=OBSERVER, frame=FRAME, orbit_path=ORBIT_PATH, save_format=SAVE_EXT)
    
    ephemerides = {}
    if CARTESIAN:
        ephemerides['cartesian'] = spice.cartesian_plot(
            SATELLITES, START, stop=STOP, td=CADENCE, plane=PLANE, sw_speed=SW_SPEED, **orbit_p)
    if POLAR:
        ephemerides['polar'] = spice.polar_plot(
            SATELLITES, START, stop=STOP, td=CADENCE, sw_speed=SW_SPEED,
            rmax=RMAX, **orbit_p)
    if FOOTPOINTS:
        footpoints = spice.ss_footpoints(
            SATELLITES, START, stop=STOP, td=CADENCE, sw_speed=SW_SPEED, **footpoint_p)
    if EPHEMERIS:
        ephemerides['system'] = spice.system_ephemeris(
            SATELLITES, START, stop=STOP, td=CADENCE)
    figures = spice.figures
    
    
    
    
    
    
    
    
    
    if figures:
        plt.show()
    # Kernels stay loaded so you can use the spice object in Spyder's console.
    # Call spice.clear_kernels() when finished (clears the global SPICE pool).
