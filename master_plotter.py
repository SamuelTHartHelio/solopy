"""Plot SIS inverse speed and MAG on a shared UTC time axis.

Edit the settings below and run this file in Spyder or with
`python master_plotter.py` from the solopy checkout.

Set EVENT to 0 to use START and STOP and show markers from overlapping
events. Otherwise, select one event from events.csv. Use columns Event,
Start, Stop, Dropout, Dropin and full UTC timestamps. Separate multiple
times in a marker cell with semicolons.
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
from .dropout_stats import extract_durations
from .sis_pha import SIS_PHA


EVENT = 36
EVENTS_FILE = Path('/Users/shart/Nextcloud/Shared/NC_Hart/solo/HCS_Dropouts/events.csv')

START = "2023-10-17 12:00"
STOP = "2023-10-19 06:00"
PHA_DATA_DIR = Path('/Users/shart/Documents/Work/SOSIS/Data/PHA/txt/')
MAG_DATA_DIR = Path('/Users/shart/Nextcloud/Shared/NC_Hart/solo/Data/')
MAG_RESOLUTION = "1min"
OUTPUT_FILE = EVENTS_FILE.parent / 'case1.png'  # Set to a path such as "sis_mag.png" to save the figure.


def plot_sis_mag(start, stop, *, pha_data_dir, mag_data_dir,
                 mag_resolution="1min", **oos_options):
    """Return the figure and axes. Extra keywords go to SIS_PHA.plot_oos_AB."""
    start, stop = pd.Timestamp(start), pd.Timestamp(stop)
    if not start < stop:
        raise ValueError("start must be earlier than stop")

    pha = SIS_PHA(start, stop, data_dir=Path(pha_data_dir).expanduser())
    mag = read_solo.load_mag(start, stop, mag_resolution, data_dir=Path(mag_data_dir).expanduser())

    fig = plt.figure(figsize=(8, 6))
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


def main(event_number):
    start, stop = START, STOP
    events = pd.read_csv(EVENTS_FILE, dtype=str, keep_default_na=False)
    if event_number == 0:
        plot_start, plot_stop = pd.to_datetime(start, utc=True), pd.to_datetime(stop, utc=True)
        event_start = pd.to_datetime(events["Start"], utc=True, format="mixed")
        event_stop = pd.to_datetime(events["Stop"], utc=True, format="mixed")
        matches = events.loc[event_start.le(plot_stop) & event_stop.ge(plot_start)]
    else:
        matches = events.loc[events["Event"].str.strip().eq(str(event_number))]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one row for event {event_number} in {EVENTS_FILE}")

        event = matches.iloc[0]
        start, stop = event["Start"], event["Stop"]
        plot_start, plot_stop = pd.to_datetime(start, utc=True), pd.to_datetime(stop, utc=True)

    pairs, review = extract_durations(matches.assign(Type=matches.get("Type", "")))
    visible_pairs = pairs.loc[pairs["Dropout"].lt(plot_stop) & pairs["Dropin"].gt(plot_start)]
    unmatched = review.loc[
        review["Reason"].ne("outside event window") & review["Time"].between(plot_start, plot_stop)
    ]

    fig, axes = plot_sis_mag(
        start, stop, pha_data_dir=PHA_DATA_DIR, mag_data_dir=MAG_DATA_DIR,
        mag_resolution=MAG_RESOLUTION, tof=[3,1], freq="8min",
        mrange=[2,80], exclude_he4=True,
    )
    for ax in axes:
        for dropout, dropin in visible_pairs[["Dropout", "Dropin"]].itertuples(index=False, name=None):
            endpoints = [max(dropout, plot_start), min(dropin, plot_stop)]
            ax.plot(
                [time.tz_convert(None) for time in endpoints], [0.95, 0.95],
                transform=ax.get_xaxis_transform(), color="dimgray", linewidth=5,
                solid_capstyle="butt", scalex=False, scaley=False, zorder=4,
            )
        for kind, time in unmatched[["Marker", "Time"]].itertuples(index=False, name=None):
            color, style = ("dimgray", "-") if kind == "Dropout" else ("gray", "--")
            ax.axvline(time.tz_convert(None), color=color, linestyle=style, linewidth=1)

    if OUTPUT_FILE is not None:
        fig.savefig(OUTPUT_FILE, dpi=300)
    plt.show()


if __name__ == "__main__":
    main(EVENT)
