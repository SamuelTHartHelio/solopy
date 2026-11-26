"""Electron Proton Telescope data from the public Solar Orbiter 1-minute CDFs."""

import re
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

if not __package__:
    # Spyder's Run File needs the directory containing the solopy checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solopy.read_solo import load_ept, load_mag
from solopy.solopy_utils import curate_axes


class EPT:
    """Load EPT data for an inclusive UTC interval at the requested cadence.

    Variables on auxiliary time axes are retained as loaded from the CDFs.
    """

    _NATIVE_CADENCE = pd.Timedelta("1min")
    _FLUX_KEY = re.compile(r"(?:Ion|Electron|Electron_Corrected)_(Flux|Uncertainty)_[SAND]")
    _ION_LOOK_RTN = np.array([[0.8191, 0.5735, 0], [-0.8191, -0.5735, 0], 
                              [-0.321, 0.458, 0.829], [0.321, -0.458, -0.829]])
    _FOV = 30

    def __init__(self, start, stop, cadence, output_dir=None, *, data_dir=None):
        self.start, self.stop = pd.to_datetime(start), pd.to_datetime(stop)
        if not self.start <= self.stop: raise ValueError("start must be on or before stop")
        self.cadence = pd.to_timedelta(cadence)
        if data_dir == None:
            self.data_dir = '/Users/shart/Nextcloud/Shared/NC_Hart/solo/Data/'
        else:
            self.data_dir = data_dir
        if self.cadence < self._NATIVE_CADENCE or self.cadence % self._NATIVE_CADENCE != pd.Timedelta(0):
            raise ValueError("EPT cadence must be a positive integer multiple of 1min")
        self.rate = "1min"
        self.factor = self.cadence // self._NATIVE_CADENCE
        self.output_dir = output_dir
        self.dat = self._load(data_dir)

    def _load(self, data_dir):
        source = load_ept(self.start, self.stop, data_dir=data_dir)
        epochs = pd.to_datetime(source['EPOCH'])
        epoch_count = len(epochs)
        selected = np.flatnonzero((epochs >= self.start) & (epochs <= self.stop))
        # Keep one record per epoch and use chronological ordering.
        selected = selected[np.argsort(epochs[selected], kind="stable")]
        selected = selected[~epochs[selected].duplicated(keep="last")]
        epochs = epochs[selected]

        if self.factor > 1:
            breaks = np.flatnonzero(epochs[1:] - epochs[:-1] != self._NATIVE_CADENCE) + 1
            runs = np.split(selected, breaks)
            groups_per_run = []
            for run in runs:
                complete_bins = len(run) // self.factor
                samples_to_keep = complete_bins * self.factor
                groups_per_run.append(run[:samples_to_keep].reshape(complete_bins, self.factor))

            groups = np.concatenate(groups_per_run)
            first = pd.to_datetime(source["EPOCH"][groups[:, 0]])
            last = pd.to_datetime(source["EPOCH"][groups[:, -1]])
            epochs = first + (last - first) / 2

        data = {"EPOCH": epochs}
        for key, values in source.items():
            if key == "EPOCH":
                continue
            # SpacePy's concatCDF copies static variables as non-record-varying VarCopy objects.
            static = hasattr(values, "rv") and not values.rv()
            dependency = getattr(values, "attrs", {}).get("DEPEND_0")
            on_main_epoch = (not key.startswith("EPOCH_") and dependency in (None, "EPOCH") and np.shape(values)[:1] == (epoch_count,))
            if static:
                data[key] = values * self.factor if key == "DELTA_EPOCH" and self.factor > 1 else values
            elif not on_main_epoch:
                data[key] = values
            elif self.factor == 1:
                data[key] = values[selected]
            elif key == "DELTA_EPOCH":
                data[key] = values[groups].sum(axis=1)
            elif match := self._FLUX_KEY.fullmatch(key):
                samples = values[groups]
                if match[1] == "Flux":
                    data[key] = self._aggregate(samples)
                    continue
                flux_key = key.replace("_Uncertainty_", "_Flux_")
                flux = source[flux_key][groups] if flux_key in source else samples
                data[key] = self._aggregate(flux, errors=samples)
        return data

    @staticmethod
    def _aggregate(samples, *, errors=None):
        """Average valid fluxes or propagate independent errors on the mean."""
        valid = np.isfinite(samples) & (samples >= 0)
        count = valid.sum(axis=1)
        if errors is None:
            total = np.where(valid, samples, 0).sum(axis=1, dtype=float)
        else:
            errors = np.where(np.isfinite(errors) & (errors >= 0), errors, np.nan)
            values = np.where(valid, errors, 0).astype(float)
            total = np.sqrt(np.square(values).sum(axis=1))
        return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)

    def summary_plot(self, species, grouped=0):
        aliases = {"ion": "Ion", "electron": "Electron", "elec": "Electron"}
        particle = aliases.get(species.strip().lower())
        if particle is None:
            raise ValueError("species must be ion, ions, electron, elec, or electrons")

        epochs = self.dat["EPOCH"]
        energy = np.asarray(self.dat[f"{particle}_Energy"]).reshape(-1)
        
        telescopes = {"S": "Sun", "A": "Anti-Sun", "N": "North", "D": "South"}
        fluxes = [self.dat[f"{particle}_Flux_{scope}"] for scope in telescopes]
        labels = list(telescopes.values())
        if grouped == 1:
            fluxes = [self._aggregate(np.stack(fluxes, axis=1))]
            labels = [f"EPT {particle.capitalize()} Telescope mean"]

        fig, axes = plt.subplots(len(fluxes), 1, sharex=True, squeeze=False, 
                                 figsize=(9, 2.5 if grouped else 7), layout="constrained")
        axes = axes[:,0]
        cmap = plt.cm.turbo.copy()
        cmap.set_bad((0,0,0))
        norm = LogNorm(vmin=1e-1, vmax=1e4)
        for ax, flux, label in zip(axes, fluxes, labels):
            mesh = ax.pcolormesh(epochs, energy, (flux * energy**2).T, shading="nearest", 
                                 cmap=cmap, norm=norm, rasterized=True,)
            ax.set_yscale("log")
            ax.set_ylabel("Energy (MeV)")
            ax.set_title(label, loc="left")

        curate_axes(axes, self.start, self.stop, year=self.start.year)
        fig.colorbar(mesh, ax=list(axes), label=r"Differential flux $\times E^2$")
        return fig, axes

    def plot_ion_pad(self, ax=None, *, energy=0.67, cg_corr=False, v_sw=400, gamma=1.5):
        "Plot ion flux at the nearest native energy channel by pitch angle."
        
        epochs = pd.DatetimeIndex(self.dat["EPOCH"])

        energies = np.asarray(self.dat["Ion_Energy"]).reshape(-1)
        channel = np.abs(energies - energy).argmin()
        channel_energy = energies[channel]
        flux = np.column_stack([self.dat[f"Ion_Flux_{scope}"][:, channel] for scope in "SAND"])
        looks = self._ION_LOOK_RTN
        looks = looks / np.linalg.norm(looks, axis=1, keepdims=True)
        if cg_corr:
            # With radial wind, the tangential sign does not affect the speed ratio.
            speed = np.sqrt(2 * 1.602e-13 * channel_energy / 1.67e-27) / 1e3
            shifted_velocity = speed * looks - [v_sw, 0, 0]
            energy_ratio = np.square(np.linalg.norm(shifted_velocity, axis=1) / speed)
            exponent = 1 if gamma == -1 else 1 + gamma
            flux = flux * energy_ratio**exponent

        half_bin = self.cadence / 2
        left_edges, right_edges = epochs - half_bin, epochs + half_bin
        mag = load_mag(left_edges[0], right_edges[-1], "1min", data_dir=self.data_dir )
        mag_times = pd.DatetimeIndex(mag["EPOCH"])
        field = np.asarray(mag["B_RTN"])[:, :3]
        magnitude = np.linalg.norm(field, axis=1)
        
        valid = mag_times.notna() & np.isfinite(magnitude) & (magnitude > 0)
        times = mag_times[valid]

        directions = (field[valid] / magnitude[valid, None])
        angles = np.degrees(np.arccos(np.clip(directions @ looks.T, -1, 1)))
        first = times.searchsorted(left_edges, side="left")
        stop = times.searchsorted(right_edges, side="left")

        vertices, colors = [], []
        for index, (start_mag, stop_mag) in enumerate(zip(first, stop)):
            if start_mag == stop_mag:
                continue
            pitch_angles = angles[start_mag:stop_mag].mean(axis=0)
            left, right = mdates.date2num([left_edges[index], right_edges[index]])
            for angle, value in zip(pitch_angles, flux[index]):
                if not np.isfinite(value) or value <= 0:
                    continue
                lower = max(0, angle - self._FOV / 2)
                upper = min(180, angle + self._FOV / 2)
                vertices.append([(left, lower), (right, lower), (right, upper), (left, upper)])
                colors.append(value)

        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(8, 3))
        mesh = PolyCollection(vertices, array=np.asarray(colors), cmap="jet",
                              norm=LogNorm(vmin=1e-1, vmax=1e4), rasterized=True)
        ax.add_collection(mesh)
        ax.xaxis_date()
        ax.set_ylim(0, 180)
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.set_ylabel(f"Ion pitch angle (°)\n{channel_energy:.2f} MeV")
        colorbar_ax = ax.inset_axes([1.02, 0.05, 0.025, 0.9])
        label = "CG-corrected differential flux" if cg_corr else "Differential flux"
        ax.figure.colorbar(mesh, cax=colorbar_ax, label=label)
        if standalone:
            curate_axes([ax], self.start, self.stop, year=self.start.year)
            ax.figure.subplots_adjust(right=0.82, bottom=0.22)
        return ax.figure, ax


if __name__ == '__main__':
    start, stop = pd.to_datetime('2022-11-14T00:00:00'), pd.to_datetime('2022-11-15T23:59:00')
    output_dir = '/Users/shart/Nextcloud/Shared/NC_Hart/solo/HCS_Dropouts/'
    ept = EPT(start, stop, cadence='10min', output_dir=output_dir)
    ept.summary_plot(species='ion',grouped=0)
    ept.plot_ion_pad(energy=0.4)
