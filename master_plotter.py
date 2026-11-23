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


START = "2024-04-01 00:00"
STOP = "2024-04-02 00:00"
PHA_DATA_DIR = Path(__file__).resolve().parent / "Data" / "PHA"
MAG_DATA_DIR = Path(__file__).resolve().parent / "Data"
MAG_RESOLUTION = "1min"
OUTPUT_FILE = None  # Set to a path such as "sis_mag.png" to save the figure.


def plot_sis_mag(start, stop, *, pha_data_dir, mag_data_dir,
                 mag_resolution="1min", **oos_options):
    """Return the figure and axes. Extra keywords go to SIS_PHA.plot_oos_AB."""
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if not start < stop:
        raise ValueError("start must be earlier than stop")

    pha = SIS_PHA(start, stop, data_dir=Path(pha_data_dir).expanduser())
    mag = read_solo.load_mag(
        start, stop, mag_resolution, data_dir=Path(mag_data_dir).expanduser()
    )

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 2, 1.5]}
    )
    pha.plot_oos_AB(axs=axes[:2], **oos_options)
    axes[1].set_xlabel("")

    ax = axes[2]
    for field, label, color in zip(
        mag["B_RTN"].T, mag["LBL1_B_RTN"], ("tab:red", "tab:green", "tab:blue", "black")
    ):
        ax.plot(mag["EPOCH"], field, label=label, color=color, linewidth=0.9)
    ax.set_ylabel("Magnetic field (nT)")
    ax.legend(loc="upper right", ncol=4)

    su.curate_axes(axes, start=start, stop=stop, year=start.year)
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.12, top=0.97, hspace=0.10)
    return fig, axes


if __name__ == "__main__":
    fig, axes = plot_sis_mag(
        START, STOP, pha_data_dir=PHA_DATA_DIR, mag_data_dir=MAG_DATA_DIR,
        mag_resolution=MAG_RESOLUTION, tof=[2, 3],
    )
    if OUTPUT_FILE is not None:
        fig.savefig(OUTPUT_FILE, dpi=300)
    plt.show()
