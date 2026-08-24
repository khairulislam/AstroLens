"""Example-only helpers for streaming the MultimodalUniverse Legacy Survey
HATS catalog (`hugging-science/mmu_legacysurvey_dr10_south_21`, 123M objects,
61 TiB) via `lsdb` — the same raw-flux catalog family AION-1 itself was
pretrained on. `lsdb`'s role here is spatial partitioning: a cone search
narrows a query region down to the handful of HEALPix (HATS) partition
files that actually cover it, instead of scanning the full catalog.

Two workarounds below are required in this environment (confirmed by hand,
not theoretical):

1. lsdb's default (threaded) dask scheduler corrupts concurrent reads
   against this dataset's remote filesystem backend (`OSError: Couldn't
   deserialize thrift: ... Invalid data`); every lsdb call here runs under
   `dask.config.set(scheduler="synchronous")`.
2. Reading the wide `image`/`rgb` columns over the remote filesystem hits
   the same corruption even single-threaded, so lsdb here only ever reads
   lightweight metadata (`ra`/`dec`/`object_id`). The actual per-object data
   is read by downloading the resolved partition file(s) with
   `huggingface_hub.hf_hub_download` (a plain HTTP GET, tens of MB each)
   and parsing them locally with `pyarrow` — still only the partitions the
   cone search identified, not the full catalog.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "hugging-science/mmu_legacysurvey_dr10_south_21"
CATALOG_URL = f"hf://datasets/{REPO_ID}"

_dataset_root = None


def open_catalog():
    """Open the HATS catalog's metadata (no partition data fetched yet)."""
    import lsdb

    return lsdb.open_catalog(CATALOG_URL, columns=["ra", "dec", "object_id"])


def cone_search_pixels(catalog, ra: float, dec: float, radius_arcsec: float) -> list:
    """Which HATS partitions (HEALPix pixels) cover a cone around `(ra, dec)`."""
    import dask

    with dask.config.set(scheduler="synchronous"):
        return catalog.cone_search(ra=ra, dec=dec, radius_arcsec=radius_arcsec).get_healpix_pixels()


def _find_dataset_root() -> str:
    """The catalog's real per-object image data folder — a `..._10arcs`
    sibling of the ra/dec-only metadata folder `lsdb.open_catalog` reads by
    default — found once via a cheap top-level repo listing."""
    global _dataset_root
    if _dataset_root is None:
        _dataset_root = next(
            item.path
            for item in HfApi().list_repo_tree(REPO_ID, repo_type="dataset")
            if item.path.endswith("_10arcs")
        )
    return _dataset_root


def download_partitions(pixels: list) -> list:
    """Download each HATS leaf partition's parquet file (tens of MB) to local cache."""
    root = _find_dataset_root()
    paths = []
    for pixel in pixels:
        directory = (pixel.pixel // 10_000) * 10_000
        filename = f"{root}/dataset/Norder={pixel.order}/Dir={directory}/Npix={pixel.pixel}.parquet"
        paths.append(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename))
    return paths


def load_table(paths: list) -> pd.DataFrame:
    """Read downloaded partition files into one table with `ra`, `dec`,
    `object_id`, `flux` (`(4, H, W)` float32, AION's expected layout) and
    `rgb` (raw PNG bytes, decode with `PIL.Image.open(io.BytesIO(...))`,
    for display only)."""

    def to_flux(img):
        return np.stack([np.stack([np.asarray(row, dtype="float32") for row in band]) for band in img["flux"]])

    frames = []
    for path in paths:
        table = pq.read_table(path, columns=["ra", "dec", "object_id", "image", "rgb"]).to_pandas()
        table["flux"] = table["image"].map(to_flux)
        table["rgb"] = table["rgb"].map(lambda x: x["bytes"])
        frames.append(table.drop(columns=["image"]))
    return pd.concat(frames, ignore_index=True)
