#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:14:27 2026

@author: shart
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codicepy.cospice import COSPICE

#------------------------------GENERAL SETTINGS-------------------------------#
KERNEL_PATH = Path.home() / 'anaconda3' / 'krnls' / 'solar_sys_pos.txt'
ORBIT_PATH = KERNEL_PATH.parent  # Contains ace_spice/ and wind_spice/ orbit CDFs
# Use a kernel or metakernel containing the selected targets and time interval.
START = '2023-01-10T00:00:00'
STOP = '2023-12-20T00:00:00'     # None for a single time
CADENCE = '1h'
EVENTS_FILE = Path('/Users/shart/Nextcloud/Shared/NC_Hart/solo/HCS_Dropouts/impulsive_events.csv')             # Set to Path('events.xlsx') or Path('events.csv')
EVENT_TARGET = 'Solar Orbiter'
EVENT_LINEWIDTH = 3

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


def load_event_intervals(path):
    """Read labeled UTC intervals and optional event types from Excel or CSV."""
    path = Path(path).expanduser()
    read_table = pd.read_csv if path.suffix.lower() == '.csv' else pd.read_excel
    table = read_table(path, usecols=lambda column: column in ('Event', 'Start', 'Stop', 'Type'),
                       dtype={'Event': str})
    dates = table[['Start', 'Stop']].dropna(how='all')
    event_numbers = table.get('Event')
    event_types = table.get('Type')
    intervals = []
    for index, start, stop in dates.itertuples(name=None):
        start, stop = (pd.to_datetime(value, utc=True, errors='coerce') for value in (start, stop))
        if pd.isna(start) or pd.isna(stop) or start >= stop:
            raise ValueError(f'{path}: spreadsheet row {index + 2} needs valid Start and Stop times with Start before Stop')
        event = event_numbers.loc[index] if event_numbers is not None else None
        label = str(event).strip() if pd.notna(event) else ''
        event_type = event_types.loc[index] if event_types is not None else None
        event_type = str(event_type).strip().casefold() if pd.notna(event_type) else ''
        intervals.append((label or str(index + 1), start, stop, event_type))
    return intervals


def highlight_orbit(ax, ephemeris, intervals, coordinates, linewidth):
    """Label each visible interval and return the skipped and clipped labels."""
    dates = pd.to_datetime(ephemeris['Date'], utc=True)
    origin = dates[0]
    seconds = (dates - origin).total_seconds().to_numpy()
    horizontal, vertical = coordinates
    x = ephemeris['Pos'][horizontal]
    y = ephemeris['Pos'][vertical]
    if horizontal == 'theta':
        x = np.unwrap(x)
    skipped, clipped = [], []
    colors = {'impulsive': 'lightblue', 'gradual': 'navy'}
    for label, start, stop, event_type in intervals:
        start_seconds = (start - origin).total_seconds()
        stop_seconds = (stop - origin).total_seconds()
        left = max(0, start_seconds)
        right = min(seconds[-1], stop_seconds)
        if left >= right:
            skipped.append(label)
            continue
        if left != start_seconds or right != stop_seconds:
            clipped.append(label)
        segment = np.r_[left, seconds[(seconds > left) & (seconds < right)], right]
        ax.plot(
            np.interp(segment, seconds, x), np.interp(segment, seconds, y),
            color=colors.get(event_type, 'lightblue'), linewidth=linewidth, zorder=2.5,
        )
        midpoint = (left + right) / 2
        ax.annotate(
            label, (np.interp(midpoint, seconds, x), np.interp(midpoint, seconds, y)),
            xytext=(0, 5), textcoords='offset points', ha='center', va='bottom',
            fontsize=8, zorder=4, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0},
        )
    return skipped, clipped


if __name__ == '__main__':
    if not (CARTESIAN or POLAR or FOOTPOINTS or EPHEMERIS):
        raise ValueError('Enable at least one COSPICE output.')
    if SAVE_FIG:
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    intervals = load_event_intervals(EVENTS_FILE) if EVENTS_FILE and (CARTESIAN or POLAR) else []
    if intervals and EVENT_TARGET not in SATELLITES:
        raise ValueError('EVENT_TARGET must be included in SATELLITES.')
    spice = COSPICE(kernel_path=KERNEL_PATH,output_dir=None,
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
    fig = figures["polar"]  # Use "polar" for the polar plot.
    fig.set_size_inches(6.5,6.5)   # Width and height in inches
    fig.tight_layout()
    fig.canvas.draw_idle()

    for kind, coordinates in (('cartesian', tuple(PLANE.lower())), ('polar', ('theta', 'r'))):
        if kind in figures:
            fig = figures[kind]
            if intervals:
                skipped, clipped = highlight_orbit(
                    fig.axes[0], ephemerides[kind][EVENT_TARGET], intervals, coordinates,
                    EVENT_LINEWIDTH,
                )
                print(f'{kind.capitalize()}: highlighted {len(intervals) - len(skipped)} of {len(intervals)} events.')
                if skipped:
                    print(f'Outside plotted ephemeris: {", ".join(skipped)}')
                if clipped:
                    print(f'Partly outside plotted ephemeris: {", ".join(clipped)}')
            if SAVE_FIG:
                fig.savefig(Path(SAVE_PATH) / ('system_ephemeris_' + kind + spice.save_format), dpi=600)
    spice.output_dir = SAVE_PATH if SAVE_FIG else None


    if figures:
        plt.show()
    # Kernels stay loaded so you can use the spice object in Spyder's console.
    # Call spice.clear_kernels() when finished (clears the global SPICE pool).
