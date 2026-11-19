# solopy
Python scripts developed with the strongly supervised help of AI. The purpose is easy visualization of Solar Orbiter data. Primarily in situ observations, but perhaps some remote observations.

## Download Solar Orbiter data

Install the downloader dependencies with `python -m pip install requests beautifulsoup4`.
From the repository directory, run:

```python
from download_solo import download_solo

files = download_solo("2024-05-01", "2024-05-03", ["mag_1min", "sis_30sec"])
```

Both dates are inclusive. Available products are `mag_4sec`, `mag_1min`,
`sis_30min`, `sis_30sec`, `sis_3sec`, `pas_moments`, `eas`, and
`ept_1min`. Each SIS product downloads data from both telescopes.

Files are saved in product subfolders under `./Data`. Pass
`data_dir="/path/to/Data"` to use an existing archive or the directory your
reader is configured to use. The function skips files already on disk and
returns the paths it downloaded.
