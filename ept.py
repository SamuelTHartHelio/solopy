"""Electron Proton Telescope data from the public Solar Orbiter 1-minute CDFs."""

import re

import numpy as np
import pandas as pd

from solopy.read_solo import load_ept


class EPT:
    """Load EPT data for an inclusive UTC interval at the requested cadence.

    `dat` uses the CDF variable names. At coarser cadences it contains
    epochs, bin widths, static metadata, and rebinned fluxes and uncertainties.
    `data_dir` is the directory containing the ept_1min subdirectory.
    """

    _NATIVE_CADENCE = pd.Timedelta("1min")
    _FLUX_KEY = re.compile(r"(?:Ion|Electron|Electron_Corrected)_(Flux|Uncertainty)_[SAND]")

    def __init__(self, start, stop, cadence, output_dir=None, *, data_dir=None):
        self.start = pd.to_datetime(start, utc=True).tz_convert(None)
        self.stop = pd.to_datetime(stop, utc=True).tz_convert(None)
        if not self.start <= self.stop:
            raise ValueError("start must be on or before stop")
        self.cadence = pd.to_timedelta(cadence)
        if (pd.isna(self.cadence) or self.cadence < self._NATIVE_CADENCE
                or self.cadence % self._NATIVE_CADENCE != pd.Timedelta(0)):
            raise ValueError("EPT cadence must be a positive integer multiple of 1min")
        self.rate = "1min"
        self.factor = self.cadence // self._NATIVE_CADENCE
        self.output_dir = output_dir
        self.dat = self._load(data_dir)

    def _load(self, data_dir):
        source = load_ept(self.start, self.stop, data_dir=data_dir)
        epochs = pd.to_datetime(source["EPOCH"], utc=True).tz_convert(None)
        selected = np.flatnonzero((epochs >= self.start) & (epochs <= self.stop))
        # Keep one record per epoch and use chronological ordering.
        selected = selected[np.argsort(epochs[selected], kind="stable")]
        selected = selected[~epochs[selected].duplicated(keep="last")]
        epochs = epochs[selected]

        if self.factor > 1:
            breaks = np.flatnonzero(epochs[1:] - epochs[:-1] != self._NATIVE_CADENCE) + 1
            runs = np.split(selected, breaks)
            groups = np.concatenate([
                run[:len(run) // self.factor * self.factor].reshape(-1, self.factor)
                for run in runs
            ])
            first = pd.to_datetime(source["EPOCH"][groups[:, 0]], utc=True).tz_convert(None)
            last = pd.to_datetime(source["EPOCH"][groups[:, -1]], utc=True).tz_convert(None)
            epochs = first + (last - first) / 2

        data = {"EPOCH": epochs}
        for key, values in source.items():
            if key == "EPOCH":
                continue
            # SpacePy's concatCDF copies static variables as non-record-varying VarCopy objects.
            static = hasattr(values, "rv") and not values.rv()
            if static:
                data[key] = values * self.factor if key == "DELTA_EPOCH" and self.factor > 1 else values
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
