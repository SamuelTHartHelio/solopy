"""Download Solar Orbiter CDF products from NASA CDAWeb."""

import re
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


ARCHIVE = "https://cdaweb.gsfc.nasa.gov/pub/data/solar-orbiter/"
dpath = "Data"
PRODUCTS = {"mag_4sec": ("mag/science/l2/rtn-normal",),
            "mag_1min": ("mag/science/l2/rtn-normal-1-minute",),
            "sis_30min": ("epd/science/l2/sis/a-rates-slow", "epd/science/l2/sis/b-rates-slow"),
            "sis_30sec": ("epd/science/l2/sis/a-rates-medium", "epd/science/l2/sis/b-rates-medium"),
            "sis_3sec": ("epd/science/l2/sis/a-rates-fast", "epd/science/l2/sis/b-rates-fast"),
            "pas_moments": ("swa/science/l2/pas-grnd-mom",),
            "eas": ("swa/science/l3/eas-nmpad-psd",),
            "ept_1min": ("epd/science/l3/ept-1min",)}
CDF_DATE = re.compile(r"_(\d{8})(?:T\d{6})?(?:-(\d{8})(?:T\d{6})?)?_V\d+\.cdf$", re.IGNORECASE)


def download_solo(start_date, stop_date, products, *, data_dir=None):
    """Download CDFs for an inclusive range of dates.

    Use YYYY-MM-DD strings or datetime.date objects for the dates.
    Products are mag_4sec, mag_1min, sis_30min, sis_30sec, sis_3sec,
    pas_moments, eas, and ept_1min. Each SIS product includes both telescopes.

    Files go into one subfolder per product under data_dir (default: ./Data).
    Existing files are skipped. Returns a list of newly downloaded Paths.
    """
    start = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    stop = date.fromisoformat(stop_date) if isinstance(stop_date, str) else stop_date
    if start > stop:
        raise ValueError("start_date must be on or before stop_date")

    if isinstance(products, str):
        products = [products]
    products = list(dict.fromkeys(products))
    unknown = [product for product in products if product not in PRODUCTS]
    if unknown:
        raise ValueError(f"Unknown data products: {unknown}. Choose from {', '.join(PRODUCTS)}")

    root = Path(dpath if data_dir is None else data_dir).expanduser()
    downloaded = []
    with requests.Session() as session:
        for product in products:
            for directory in PRODUCTS[product]:
                for year in range(start.year, stop.year + 1):
                    url = f"{ARCHIVE}{directory}/{year}/"
                    response = session.get(url, timeout=30)
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    for link in soup.find_all("a", href=True):
                        name = link["href"]
                        if Path(name).name != name:
                            continue
                        match = CDF_DATE.search(name)
                        if not match:
                            continue
                        first, last = match.groups()
                        if date.fromisoformat(first) > stop or date.fromisoformat(last or first) < start:
                            continue
                        destination = root / product / name
                        if destination.is_file():
                            continue
                        _save_cdf(session, urljoin(url, name), destination)
                        downloaded.append(destination)
    return downloaded


def _save_cdf(session, url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    temporary.write(chunk)
                temporary.close()
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)


def _download_recent(products):
    # The legacy helpers used the same recent-year range for either update value.
    year = date.today().year
    download_solo(date(year - 1, 1, 1), date(year, 12, 31), products)


def dl_solo_mag(resolution, update=0):
    _download_recent([f"mag_{resolution}"])


def dl_solo_sis(resolution="all", update=0):
    products = {"slow": "sis_30min", "medium": "sis_30sec", "fast": "sis_3sec"}
    if resolution != "all" and resolution not in products:
        raise ValueError("SIS resolution must be all, slow, medium, or fast")
    _download_recent(products.values() if resolution == "all" else [products[resolution]])


def dl_solo_pas_moments(update=0):
    _download_recent(["pas_moments"])


def dl_solo_eas(update=0):
    _download_recent(["eas"])


def dl_solo_ept(update=0):
    _download_recent(["ept_1min"])
