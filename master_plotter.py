"""Plot SIS inverse speed and MAG on a shared UTC time axis.

Edit the settings below and run this file in Spyder or with
`python master_plotter.py` from the solopy checkout.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if not __package__:
    # Direct execution (including Spyder Run File) needs the package parent.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "solopy"

from . import read_solo
from . import solopy_utils as su
from .sis_pha import SIS_PHA


START = "2023-01-11 12:00"
STOP = "2023-01-13 18:00"
PHA_DATA_DIR = Path('/Users/shart/Documents/Work/SOSIS/Data/PHA/txt/')
MAG_DATA_DIR = Path('/Users/shart/Nextcloud/Shared/NC_Hart/solo/Data/')
MAG_RESOLUTION = "1min"
OUTPUT_FILE = None  # Set to a path such as "sis_mag.png" to save the figure.


def plot_sis_mag(start, stop, *, pha_data_dir, mag_data_dir,
                 mag_resolution="1min", **oos_options):
    """Return the figure and axes. Extra keywords go to SIS_PHA.plot_oos_AB."""
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if not start < stop:
        raise ValueError("start must be earlier than stop")

    pha = SIS_PHA(start, stop, data_dir=Path(pha_data_dir).expanduser())
    mag = read_solo.load_mag(start, stop, mag_resolution, data_dir=Path(mag_data_dir).expanduser())

    fig = plt.figure(figsize=(9, 6))
    axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})

    pha.plot_oos_AB(axs=axs[:2], **oos_options)

    ax = axs[2]
    colors = ['tomato','darkgreen','blue','black']
    for i in range(4):
        ax.plot(mag["EPOCH"], mag['B_RTN'][:,i], color=colors[i], linewidth=0.9)
    ax.set_ylabel("Magnetic field (nT)",fontsize=8)

    su.curate_axes(axs, start=start, stop=stop, year=start.year)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, axs


if __name__ == "__main__":
    fig, axes = plot_sis_mag(
        START, STOP, pha_data_dir=PHA_DATA_DIR, mag_data_dir=MAG_DATA_DIR, mag_resolution=MAG_RESOLUTION, 
        tof=[2, 3], freq='8min',mrange=[2,80],exclude_he4=True)
    if OUTPUT_FILE is not None:
        fig.savefig(OUTPUT_FILE, dpi=300)
    plt.show()
