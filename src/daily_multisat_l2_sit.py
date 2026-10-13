#!/usr/bin/env python3
"""Daily Russian-Arctic sea-ice thickness from Level-2 altimetry.

Inputs:
- Sentinel-3A/B SRAL SR_2_LAN_SI Level-2
- CryoSat-2 Level-2 NetCDF (local ingestion)

No Level-3/Level-4 sea-ice-thickness product is used as input.
The output SIT is recomputed from Level-2 freeboard, snow and density fields.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import time as time_module
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
OUTPUT_CRS = CRS.from_epsg(3413)
RHO_W = 1024.0
# Timeliness preference for duplicate Sentinel-3 versions of the same pass:
# non-time-critical is the best-quality reprocessing, near-real-time the worst.
TIMELINESS_RANK = {"NT": 3, "ST": 2, "NR": 1}
TOKEN_REFRESH_MARGIN_S = 60.0
MIN_LAT = 65.0
MAX_LAT = 90.0
MIN_LON_360 = 30.0
MAX_LON_360 = 190.0


@dataclass
class Product:
    id: str
    name: str
    start: str | None = None
    end: str | None = None


def parse_date(value: str) -> datetime:
    if len(value) == 10:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def lon360(lon: np.ndarray) -> np.ndarray:
    return (lon % 360.0 + 360.0) % 360.0


def russian_arctic_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    x = lon360(lon)
    return (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= MIN_LAT)
        & (lat <= MAX_LAT)
        & (x >= MIN_LON_360)
        & (x <= MAX_LON_360)
    )


class CdseAuth:
    """CDSE bearer token that is refreshed before it expires.

    CDSE access tokens live about 10 minutes, which is shorter than a 7-day
    download, so the token is renewed with the refresh token when needed.
    """

    def __init__(self) -> None:
        self._access: str | None = None
        self._refresh: str | None = None
        self._expires_at = 0.0
        self._static = os.getenv("CDSE_ACCESS_TOKEN")
        if self._static:
            self._access = self._static.strip()
            self._expires_at = float("inf")

    def _store(self, payload: dict) -> None:
        self._access = payload["access_token"]
        self._refresh = payload.get("refresh_token")
        self._expires_at = time_module.monotonic() + float(payload.get("expires_in", 600))

    def _login(self) -> None:
        username = os.getenv("CDSE_USERNAME") or input("CDSE username: ").strip()
        password = os.getenv("CDSE_PASSWORD") or getpass.getpass("CDSE password: ")
        data = {
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        }
        totp = os.getenv("CDSE_TOTP")
        if totp:
            data["totp"] = totp
        response = requests.post(TOKEN_URL, data=data, timeout=60)
        if response.status_code == 401:
            raise RuntimeError(
                "CDSE authentication failed (401). Check email/password and, if 2FA is enabled, set CDSE_TOTP."
            )
        response.raise_for_status()
        self._store(response.json())

    def _refresh_access(self) -> None:
        if self._refresh:
            response = requests.post(
                TOKEN_URL,
                data={"client_id": "cdse-public", "grant_type": "refresh_token", "refresh_token": self._refresh},
                timeout=60,
            )
            if response.ok:
                self._store(response.json())
                return
        # Refresh token expired or missing: log in again.
        self._login()

    def token(self, force_refresh: bool = False) -> str:
        if self._static:
            if force_refresh:
                raise RuntimeError("CDSE_ACCESS_TOKEN was rejected or expired; unset it to log in with a password.")
            return self._access
        if self._access is None:
            self._login()
        elif force_refresh or time_module.monotonic() > self._expires_at - TOKEN_REFRESH_MARGIN_S:
            self._refresh_access()
        return self._access


def product_filter(start: datetime, end: datetime) -> str:
    start_s = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_s = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Technical Russian-Arctic AOI split at the antimeridian.
    aoi_west = "POLYGON((30 65,179.9 65,179.9 89.9,30 89.9,30 65))"
    aoi_east = "POLYGON((-180 65,-170 65,-170 89.9,-180 89.9,-180 65))"
    spatial = (
        "(OData.CSC.Intersects(area=geography'SRID=4326;" + aoi_west + "') or "
        "OData.CSC.Intersects(area=geography'SRID=4326;" + aoi_east + "'))"
    )

    return (
        "Collection/Name eq 'SENTINEL-3' and "
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'SR_2_LAN_SI') and "
        f"{spatial} and "
        f"ContentDate/Start ge {start_s} and ContentDate/Start lt {end_s}"
    )


def search_s3_products(start: datetime, end: datetime) -> list[Product]:
    params = {
        "$filter": product_filter(start, end),
        "$select": "Id,Name,ContentDate",
        "$orderby": "ContentDate/Start asc",
        "$top": "1000",
    }
    url = ODATA
    products: list[Product] = []
    while url:
        response = requests.get(url, params=params if url == ODATA else None, timeout=90)
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("value", []):
            cd = item.get("ContentDate") or {}
            products.append(Product(item["Id"], item["Name"], cd.get("Start"), cd.get("End")))
        url = payload.get("@odata.nextLink")
        params = None
    return choose_s3_timeliness(products)


S3_NAME_RE = re.compile(
    r"^(?P<mission>S3[AB_])_SR_2_LAN_SI_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<created>\d{8}T\d{6})_"
    r"(?P<instance>.{17})_(?P<centre>.{3})_(?P<platform>.)_(?P<timeliness>NR|ST|NT)_(?P<baseline>\d{3})"
)


def s3_pass_key(name: str) -> tuple[tuple, tuple] | None:
    """Return (pass identity, preference) for a Sentinel-3 product name.

    The pass is identified by mission, cycle and relative orbit from the
    instance id (``DDDD_CCC_LLL_FFFF``); if those are blank, the sensing
    start time is used instead. Preference ranks NT > ST > NR, then the
    newest baseline, then the newest creation time.
    """
    match = S3_NAME_RE.match(Path(name).name)
    if not match:
        return None
    parts = match["instance"].strip("_").split("_")
    if len(parts) >= 3 and all(p.isdigit() for p in parts[1:3]):
        identity = (match["mission"], parts[1], parts[2])
    else:
        identity = (match["mission"], match["start"])
    preference = (TIMELINESS_RANK[match["timeliness"]], match["baseline"], match["created"])
    return identity, preference


def choose_s3_timeliness(items: list, name=lambda item: item.name) -> list:
    """Keep one version per Sentinel-3 pass to avoid double counting.

    NR/ST/NT versions and reprocessing baselines of one pass overlap in time
    and space. For each pass the best version is kept (NT > ST > NR, then the
    newest baseline). Items whose name cannot be parsed are kept as they are.
    """
    best: dict[tuple, tuple[tuple, object]] = {}
    unparsed = []
    for item in items:
        key = s3_pass_key(name(item))
        if key is None:
            unparsed.append(item)
            continue
        identity, preference = key
        if identity not in best or preference > best[identity][0]:
            best[identity] = (preference, item)
    chosen = {id(item) for _, item in best.values()}
    return [item for item in items if id(item) in chosen] + unparsed


def fetch_s3_archive(product: Product, archive: Path, auth: CdseAuth) -> None:
    """Download into a temporary ``.part`` file and rename only when complete."""
    url = f"{DOWNLOAD}({product.id})/$value"
    partial = archive.with_name(archive.name + ".part")
    for attempt in range(2):
        headers = {"Authorization": f"Bearer {auth.token(force_refresh=attempt > 0)}"}
        with requests.get(url, headers=headers, stream=True, timeout=600) as response:
            if response.status_code == 401 and attempt == 0:
                continue
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.replace(archive)
        return


def download_s3_product(product: Product, target_dir: Path, auth: CdseAuth) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = product.name[:-5] if product.name.endswith(".SEN3") else product.name
    product_dir = target_dir / f"{stem}.SEN3"
    if product_dir.exists():
        return product_dir

    archive = target_dir / f"{stem}.zip"
    if not archive.exists():
        fetch_s3_archive(product, archive, auth)
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        # A corrupt archive left by an older run: fetch it again once.
        archive.unlink()
        fetch_s3_archive(product, archive, auth)
        zf = zipfile.ZipFile(archive)
    with zf:
        zf.extractall(target_dir)

    if product_dir.exists():
        return product_dir
    candidates = [p for p in target_dir.rglob("*.SEN3") if stem in p.name]
    if not candidates:
        raise FileNotFoundError(f"Extracted .SEN3 directory not found for {product.name}")
    return candidates[0]


def first_name(ds: xr.Dataset, aliases: list[str], required: bool = True) -> str | None:
    for name in aliases:
        if name in ds.variables or name in ds.coords:
            return name
    if required:
        raise KeyError(f"None of variable aliases found: {aliases}")
    return None


def array(ds: xr.Dataset, aliases: list[str], required: bool = True) -> np.ndarray | None:
    name = first_name(ds, aliases, required)
    return None if name is None else np.asarray(ds[name].values)


def time_array(ds: xr.Dataset, aliases: list[str]) -> pd.DatetimeIndex:
    name = first_name(ds, aliases, True)
    return pd.to_datetime(np.asarray(ds[name].values), utc=True)


def normalise_concentration(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    c = np.asarray(values, dtype=float)
    finite = c[np.isfinite(c)]
    if finite.size and np.nanmax(finite) <= 1.5:
        c = c * 100.0
    return c


def compute_sit(freeboard, snow_depth, snow_density, ice_density) -> np.ndarray:
    denominator = RHO_W - ice_density
    with np.errstate(divide="ignore", invalid="ignore"):
        sit = (RHO_W * freeboard + snow_density * snow_depth) / denominator
    sit = np.where((denominator > 0) & (sit >= 0.0) & (sit <= 15.0), sit, np.nan)
    return sit


def radar_to_ice_freeboard(radar_freeboard, snow_depth, snow_density) -> np.ndarray:
    """Convert radar freeboard to sea-ice freeboard.

    The Ku-band echo travels slower in snow, so the radar freeboard is lower
    than the ice freeboard: fi = fr + hs * (c / cs - 1), with
    c / cs = (1 + 0.51 * rho_s / 1000) ** 1.5 (Ulaby et al., 1986).
    """
    rho_s = np.asarray(snow_density, dtype=float)
    ratio = (1.0 + 0.51 * rho_s / 1000.0) ** 1.5
    return np.asarray(radar_freeboard, dtype=float) + np.asarray(snow_depth, dtype=float) * (ratio - 1.0)


def is_sea_ice_class(label: str) -> bool:
    return label == "sea_ice" or "floe" in label


def sea_ice_class_values(attrs: dict) -> list[int] | None:
    """Return the surface-class codes that mean sea-ice floe, from CF flag attributes."""
    values = attrs.get("flag_values")
    meanings = attrs.get("flag_meanings")
    if values is None or not meanings:
        return None
    codes = np.atleast_1d(np.asarray(values)).tolist()
    names = str(meanings).lower().split()
    if len(codes) != len(names):
        return None
    selected = [int(code) for code, label in zip(codes, names) if is_sea_ice_class(label)]
    return selected or None


def make_frame(
    *,
    time,
    lat,
    lon,
    freeboard,
    snow_depth,
    snow_density,
    ice_density,
    source: str,
    sensor: str,
    concentration=None,
    surface_class=None,
    sea_ice_classes: list[int] | None = None,
    official_sit=None,
    radar_freeboard=None,
    freeboard_source: str = "sea_ice_freeboard",
) -> pd.DataFrame:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    shape = lat.shape
    # Some L2 products store densities as a scalar per file.
    freeboard = np.broadcast_to(np.asarray(freeboard, dtype=float), shape)
    snow_depth = np.broadcast_to(np.asarray(snow_depth, dtype=float), shape)
    snow_density = np.broadcast_to(np.asarray(snow_density, dtype=float), shape)
    ice_density = np.broadcast_to(np.asarray(ice_density, dtype=float), shape)
    concentration = normalise_concentration(concentration)

    # Conservative physical limits before the hydrostatic calculation.
    freeboard = np.where((freeboard >= -1.0) & (freeboard <= 3.0), freeboard, np.nan)
    snow_depth = np.where((snow_depth >= 0.0) & (snow_depth <= 3.0), snow_depth, np.nan)
    sit = compute_sit(freeboard, snow_depth, snow_density, ice_density)

    valid = russian_arctic_mask(lat, lon)
    valid &= np.isfinite(sit)
    valid &= np.isfinite(freeboard)
    valid &= np.isfinite(snow_depth)
    valid &= np.isfinite(snow_density)
    valid &= np.isfinite(ice_density)

    if concentration is not None:
        valid &= np.isfinite(concentration) & (concentration >= 15.0)
    if surface_class is not None and sea_ice_classes is not None:
        valid &= np.isin(np.asarray(surface_class), sea_ice_classes)

    data = {
        "time": pd.DatetimeIndex(time)[valid],
        "lat": lat[valid],
        "lon": lon[valid],
        "sit_m": sit[valid],
        "sea_ice_freeboard_m": freeboard[valid],
        "snow_depth_m": snow_depth[valid],
        "snow_density_kg_m3": snow_density[valid],
        "ice_density_kg_m3": ice_density[valid],
        "source": source,
        "sensor": sensor,
        "freeboard_source": freeboard_source,
    }
    if concentration is not None:
        data["sea_ice_concentration_pct"] = concentration[valid]
    if official_sit is not None:
        data["official_l2_sit_m_validation_only"] = np.asarray(official_sit, dtype=float)[valid]
    if radar_freeboard is not None:
        data["radar_freeboard_m"] = np.asarray(radar_freeboard, dtype=float)[valid]
    return pd.DataFrame(data)


def find_s3_nc(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".nc":
        return path
    for candidate in ("standard_measurement.nc", "enhanced_measurement.nc"):
        found = list(path.rglob(candidate))
        if found:
            return found[0]
    files = list(path.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF in {path}")
    return files[0]


def read_s3_l2(path: Path) -> pd.DataFrame:
    nc = find_s3_nc(path)
    with xr.open_dataset(nc, decode_cf=True, mask_and_scale=True) as ds:
        lat = array(ds, ["lat_20_ku"]).astype(float)
        lon = array(ds, ["lon_20_ku"]).astype(float)
        time = time_array(ds, ["time_20_ku"])

        # These aliases match the real SR_2_LAN_SI files observed in this project.
        snow_depth = array(ds, ["snow_depth_sol1_20_ku", "snow_depth_sol2_20_ku"]).astype(float)
        snow_density = array(ds, ["snow_density_20_ku"]).astype(float)
        ice_density = array(ds, ["sea_ice_density_20_ku", "ice_density_20_ku"]).astype(float)
        radar_freeboard = array(ds, ["radar_freeboard_20_ku"], required=False)
        sea_ice_freeboard = array(ds, ["sea_ice_freeboard_20_ku"], required=False)
        if sea_ice_freeboard is not None:
            freeboard = sea_ice_freeboard.astype(float)
            freeboard_source = "sea_ice_freeboard"
        elif radar_freeboard is not None:
            # Radar freeboard is not ice freeboard: correct for wave speed in snow.
            freeboard = radar_to_ice_freeboard(radar_freeboard, snow_depth, snow_density)
            freeboard_source = "radar_freeboard_corrected"
        else:
            raise KeyError("Neither sea_ice_freeboard_20_ku nor radar_freeboard_20_ku found")

        concentration = array(ds, ["sea_ice_concentration_20_ku"], required=False)
        # Only the sea-ice classification is used. The generic surf_type_20_ku
        # (open ocean / enclosed seas / continental ice / land) is not a floe flag.
        surface_class = None
        sea_ice_classes = None
        if "surf_type_class_20_ku" in ds.variables:
            surface_class = np.asarray(ds["surf_type_class_20_ku"].values)
            sea_ice_classes = sea_ice_class_values(ds["surf_type_class_20_ku"].attrs)
            if sea_ice_classes is None:
                print(
                    f"[warn s3] {path.name}: surf_type_class_20_ku has no usable flag_meanings; "
                    "surface-class filter not applied",
                    file=sys.stderr,
                )
        official_sit = array(ds, ["sea_ice_thickness_20_ku"], required=False)

    return make_frame(
        time=time,
        lat=lat,
        lon=lon,
        freeboard=freeboard,
        snow_depth=snow_depth,
        snow_density=snow_density,
        ice_density=ice_density,
        source=path.name,
        sensor="Sentinel-3",
        concentration=concentration,
        surface_class=surface_class,
        sea_ice_classes=sea_ice_classes,
        official_sit=official_sit,
        radar_freeboard=radar_freeboard,
        freeboard_source=freeboard_source,
    )


def find_cs2_nc(path: Path) -> Path:
    if path.is_file() and path.suffix.lower() == ".nc":
        return path
    files = list(path.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF in {path}")
    return files[0]


def read_cs2_l2(path: Path) -> pd.DataFrame:
    """Alias-based CryoSat-2 Level-2 ingestion.

    Different CryoSat-2 L2 distributions use different variable names. Files
    missing the Level-2 fields needed by the hydrostatic calculation are skipped.
    """
    nc = find_cs2_nc(path)
    with xr.open_dataset(nc, decode_cf=True, mask_and_scale=True) as ds:
        lat = array(ds, ["lat_20_ku", "lat_poca_20_ku", "latitude", "lat"]).astype(float)
        lon = array(ds, ["lon_20_ku", "lon_poca_20_ku", "longitude", "lon"]).astype(float)
        time = time_array(ds, ["time_20_ku", "time_poca_20_ku", "time"])
        freeboard = array(ds, ["freeboard_20_ku", "sea_ice_freeboard_20_ku", "freeboard"]).astype(float)
        snow_depth = array(ds, ["snow_depth_20_ku", "snow_depth", "snow_depth_01"]).astype(float)
        snow_density = array(ds, ["snow_density_20_ku", "snow_density"]).astype(float)
        ice_density = array(ds, ["sea_ice_density_20_ku", "ice_density_20_ku", "ice_density"]).astype(float)
        concentration = array(ds, ["sea_ice_concentration_20_ku", "sea_ice_concentration", "ice_conc"], required=False)
        official_sit = array(ds, ["sea_ice_thickness_20_ku", "sea_ice_thickness", "thickness_20_ku"], required=False)

    return make_frame(
        time=time,
        lat=lat,
        lon=lon,
        freeboard=freeboard,
        snow_depth=snow_depth,
        snow_density=snow_density,
        ice_density=ice_density,
        source=path.name,
        sensor="CryoSat-2",
        concentration=concentration,
        official_sit=official_sit,
    )


def discover_s3(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    products = sorted(raw_dir.glob("*.SEN3"))
    products += sorted(raw_dir.glob("*.nc"))
    # A local archive can hold NR, ST and NT copies of the same pass.
    return choose_s3_timeliness(products)


def discover_cs2(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    files = sorted(raw_dir.glob("*.nc"))
    files += sorted(p for p in raw_dir.iterdir() if p.is_dir() and list(p.rglob("*.nc")))
    return files


def load_frames(paths: list[Path], reader, start: datetime, target: datetime, label: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = reader(path)
        except Exception as exc:
            print(f"[skip {label}] {path}: {exc}", file=sys.stderr)
            continue
        frame = frame[(frame["time"] >= pd.Timestamp(start)) & (frame["time"] <= pd.Timestamp(target))]
        if not frame.empty:
            frames.append(frame)
    return frames


def project_points(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs(4326, OUTPUT_CRS, always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    out = df.copy()
    out["x"] = x
    out["y"] = y
    return out


def cell_stats(df: pd.DataFrame, target: datetime, grid_m: int, tau_days: float) -> pd.DataFrame:
    points = project_points(df)
    age_days = (pd.Timestamp(target) - points["time"]).dt.total_seconds() / 86400.0
    points["age_days"] = age_days.clip(lower=0)
    points["weight"] = np.exp(-points["age_days"] / tau_days)
    points["ix"] = np.floor(points["x"] / grid_m).astype(np.int64)
    points["iy"] = np.floor(points["y"] / grid_m).astype(np.int64)

    rows = []
    for (ix, iy), group in points.groupby(["ix", "iy"], sort=False):
        v = group["sit_m"].to_numpy(float)
        w = group["weight"].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not ok.any():
            continue
        v = v[ok]
        w = w[ok]
        newest = group.loc[ok, "time"].max()
        counts = group.loc[ok, "sensor"].value_counts().to_dict()
        rows.append(
            {
                "ix": int(ix),
                "iy": int(iy),
                "sit_m": float(np.sum(v * w) / np.sum(w)),
                "sit_median_m": float(np.nanmedian(v)),
                "sit_std_m": float(np.nanstd(v)),
                "n_obs": int(len(v)),
                "age_hours": max(0.0, (pd.Timestamp(target) - newest).total_seconds() / 3600.0),
                "n_s3": int(counts.get("Sentinel-3", 0)),
                "n_cs2": int(counts.get("CryoSat-2", 0)),
            }
        )
    return pd.DataFrame(rows)


def raster_geometry(cells: pd.DataFrame, grid_m: int):
    if cells.empty:
        raise RuntimeError("No observed grid cells after filtering")
    min_ix = int(cells.ix.min())
    max_ix = int(cells.ix.max())
    min_iy = int(cells.iy.min())
    max_iy = int(cells.iy.max())
    width = max_ix - min_ix + 1
    height = max_iy - min_iy + 1
    transform = from_origin(min_ix * grid_m, (max_iy + 1) * grid_m, grid_m, grid_m)
    return min_ix, max_iy, width, height, transform


def write_raster(cells: pd.DataFrame, column: str, path: Path, grid_m: int) -> None:
    min_ix, max_iy, width, height, transform = raster_geometry(cells, grid_m)
    nodata = -9999.0
    data = np.full((height, width), nodata, dtype=np.float32)
    for row in cells.itertuples(index=False):
        value = getattr(row, column)
        if np.isfinite(value):
            data[max_iy - int(row.iy), int(row.ix) - min_ix] = float(value)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=OUTPUT_CRS,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, column)
        dst.update_tags(input_level="Level-2", interpolation="none", grid_m=str(grid_m))


def write_png(cells: pd.DataFrame, path: Path, grid_m: int, target: datetime) -> None:
    min_ix, max_iy, width, height, transform = raster_geometry(cells, grid_m)
    data = np.full((height, width), np.nan, dtype=np.float32)
    for row in cells.itertuples(index=False):
        data[max_iy - int(row.iy), int(row.ix) - min_ix] = row.sit_m
    left = transform.c
    top = transform.f
    right = left + width * grid_m
    bottom = top - height * grid_m

    fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
    image = ax.imshow(
        data,
        extent=(left, right, bottom, top),
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=5,
    )
    ax.set_aspect("equal")
    ax.set_title(
        f"Sentinel-3 + CryoSat-2 Level-2 sea-ice thickness\n"
        f"{target:%Y-%m-%d} UTC | observed {grid_m / 1000:g} km cells"
    )
    ax.set_xlabel("EPSG:3413 x, m")
    ax.set_ylabel("EPSG:3413 y, m")
    cb = fig.colorbar(image, ax=ax, shrink=0.82)
    cb.set_label("Sea-ice thickness, m")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def validation_stats(obs: pd.DataFrame) -> dict:
    name = "official_l2_sit_m_validation_only"
    if name not in obs:
        return {}
    a = obs["sit_m"].to_numpy(float)
    b = obs[name].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return {}
    d = a[ok] - b[ok]
    result = {
        "n": int(ok.sum()),
        "bias_m": float(np.mean(d)),
        "mae_m": float(np.mean(np.abs(d))),
        "rmse_m": float(np.sqrt(np.mean(d**2))),
    }
    if ok.sum() > 1:
        result["correlation"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return result


def build_product(
    s3_paths: list[Path],
    cs2_paths: list[Path],
    output_dir: Path,
    target: datetime,
    days: int,
    grid_m: int,
    tau_days: float,
) -> None:
    start = target - timedelta(days=days)
    frames = load_frames(s3_paths, read_s3_l2, start, target, "s3")
    frames += load_frames(cs2_paths, read_cs2_l2, start, target, "cryosat-2")
    if not frames:
        raise RuntimeError("No valid Level-2 observations found for Sentinel-3 or CryoSat-2")

    obs = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    cells = cell_stats(obs, target, grid_m, tau_days)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = target.strftime("%Y%m%d")

    obs.to_csv(output_dir / f"multisat_l2_sit_observations_{stamp}.csv.gz", index=False, compression="gzip")
    cells.to_csv(output_dir / f"multisat_l2_sit_cells_{grid_m}m_{stamp}.csv", index=False)
    write_raster(cells, "sit_m", output_dir / f"multisat_l2_sit_{grid_m}m_{stamp}.tif", grid_m)
    write_raster(cells, "age_hours", output_dir / f"multisat_l2_observation_age_hours_{grid_m}m_{stamp}.tif", grid_m)
    write_raster(cells, "n_obs", output_dir / f"multisat_l2_observation_count_{grid_m}m_{stamp}.tif", grid_m)
    write_raster(cells, "sit_std_m", output_dir / f"multisat_l2_sit_std_{grid_m}m_{stamp}.tif", grid_m)
    write_raster(cells, "n_s3", output_dir / f"multisat_l2_s3_count_{grid_m}m_{stamp}.tif", grid_m)
    write_raster(cells, "n_cs2", output_dir / f"multisat_l2_cs2_count_{grid_m}m_{stamp}.tif", grid_m)
    write_png(cells, output_dir / f"multisat_l2_sit_{grid_m}m_{stamp}.png", grid_m, target)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target_time_utc": target.isoformat(),
        "rolling_window_days": days,
        "grid_m": grid_m,
        "source_level": "Level-2",
        "sensors": sorted(obs.sensor.unique().tolist()),
        "n_valid_observations": int(len(obs)),
        "n_observed_grid_cells": int(len(cells)),
        "n_by_sensor": {k: int(v) for k, v in obs.sensor.value_counts().to_dict().items()},
        "validation_against_official_l2_sit": validation_stats(obs),
        "method": (
            "SIT=(rho_water*sea_ice_freeboard + snow_density*snow_depth)/(rho_water-ice_density), "
            "rho_water=1024 kg/m3; official L2 SIT is validation-only"
        ),
        "spatial_interpolation": "none",
        "technical_aoi": {"lat": [65, 90], "lon_0_360": [30, 190]},
    }
    (output_dir / f"multisat_l2_sit_metadata_{stamp}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Valid L2 observations: {len(obs):,}")
    print(obs.sensor.value_counts().to_string())
    print(f"Observed {grid_m / 1000:g}-km cells: {len(cells):,}")
    print(f"Output: {output_dir}")


def run(args) -> None:
    target = parse_date(args.date)
    if target.hour == 0 and target.minute == 0:
        target = target.replace(hour=23, minute=59, second=59)

    s3_dir = Path(args.s3_raw_dir)
    cs2_dir = Path(args.cs2_raw_dir)
    output_dir = Path(args.output_dir)

    s3_paths = discover_s3(s3_dir)
    if args.s3_download:
        start = target - timedelta(days=args.days)
        print(f"Searching Sentinel-3 SR_2_LAN_SI: {start.isoformat()} .. {target.isoformat()}")
        products = search_s3_products(start, target + timedelta(seconds=1))
        print(f"Sentinel-3 Russian-Arctic products: {len(products)}")
        if products:
            auth = CdseAuth()
            s3_paths = []
            for index, product in enumerate(products, 1):
                print(f"[S3 {index}/{len(products)}] {product.name}")
                try:
                    s3_paths.append(download_s3_product(product, s3_dir, auth))
                except Exception as exc:
                    print(f"[download failed] {product.name}: {exc}", file=sys.stderr)

    cs2_paths = discover_cs2(cs2_dir)
    print(f"Sentinel-3 local inputs: {len(s3_paths)}")
    print(f"CryoSat-2 local inputs: {len(cs2_paths)}")
    build_product(s3_paths, cs2_paths, output_dir, target, args.days, args.grid_m, args.tau_days)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily Russian-Arctic SIT from Sentinel-3 + CryoSat-2 Level-2")
    p.add_argument("--date", required=True, help="UTC date/time, e.g. 2026-09-15")
    p.add_argument("--days", type=int, default=7, help="Rolling window in days")
    p.add_argument("--grid-m", type=int, default=1000, help="Observed-cell output grid in metres")
    p.add_argument("--tau-days", type=float, default=2.0, help="Temporal exponential weighting scale")
    p.add_argument("--s3-raw-dir", default="data/raw/sentinel3")
    p.add_argument("--cs2-raw-dir", default="data/raw/cryosat2")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--s3-download", action="store_true", help="Download Sentinel-3 L2 from CDSE before processing")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.grid_m < 300:
        raise SystemExit("--grid-m below 300 m is not justified for these altimetry observations")
    run(args)


if __name__ == "__main__":
    main()
