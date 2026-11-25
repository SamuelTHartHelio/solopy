"""Electron Proton Telescope data from the public Solar Orbiter 1-minute CDFs."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from solopy.read_solo import load_ept
from solopy.solopy_utils import curate_axes


class EPT:
    """Load EPT data for an inclusive UTC interval at the requested cadence.

    Variables on auxiliary time axes are retained as loaded from the CDFs.
    """

    _NATIVE_CADENCE = pd.Timedelta("1min")
    _FLUX_KEY = re.compile(r"(?:Ion|Electron|Electron_Corrected)_(Flux|Uncertainty)_[SAND]")

    def __init__(self, start, stop, cadence, output_dir=None, *, data_dir=None):
        self.start, self.stop = pd.to_datetime(start), pd.to_datetime(stop)
        if not self.start <= self.stop: raise ValueError("start must be on or before stop")
        self.cadence = pd.to_timedelta(cadence)
        if data_dir == None:
            self.data_dir = '/Users/shart/Nextcloud/Shared/NC_Hart/solo/Data/'
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
        aliases = {"ion": "Ion", "ions": "Ion", "electron": "Electron", "elec": "Electron", "electrons": "Electron"}
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
            labels = ["Telescope mean"]

        fig, axes = plt.subplots(len(fluxes), 1, sharex=True, squeeze=False, 
                                 figsize=(8, 3 if grouped else 7), layout="constrained")
        axes = axes[:,0]
        ax = axes[0]
        norm = LogNorm(vmin=1e-1, vmax=1e4)
        for ax, flux, label in zip(axes, fluxes, labels):
            mesh = ax.pcolormesh(epochs, energy, (fluxes*energy**2).T, cmap="jet", norm=norm, rasterized=True)
            ax.set_yscale("log")
            ax.set_ylabel("Energy (MeV)")
            ax.set_title(telescopes, loc="left")

        curate_axes(axes, self.start, self.stop, year=self.start.year)
        fig.suptitle(f"EPT {particle.lower()} spectrogram")
        fig.colorbar(mesh, ax=list(axes), label="Differential flux")
        return fig, axes


if __name__ == '__main__':
    start, stop = pd.to_datetime('2025-05-16T02:00:00'), pd.to_datetime('2025-05-16T16:00:00')
    output_dir = '/Users/shart/Nextcloud/Shared/NC_Hart/solo/HCS_Dropouts/'
    ept = EPT(start, stop, cadence='1min', output_dir=output_dir)
    ept.summary_plot('ion',grouped=1)
