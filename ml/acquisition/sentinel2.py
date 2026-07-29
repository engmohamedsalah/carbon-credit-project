"""Fetch a model-ready Sentinel-2 L2A stack for a project boundary from public AWS COGs.

Produces the exact input the forest-cover U-Net expects (see
docs/ml-forest-model-investigation.md): 12 bands in TRAINING_BAND_ORDER, RAW DN, on a
common 10 m grid, windowed to the boundary's bounding box. Uses the free Element84
Earth Search STAC + the open ``sentinel-2-l2a`` COGs on AWS — no Copernicus login.

This is the "boundary -> imagery" link of the analysis pipeline, and it fulfils the
band-order contract that ml/inference/production_inference.py documents but cannot
enforce on an arbitrary uploaded file.
"""
import json
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds, transform as window_transform

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"

# The exact bands + order the forest U-Net was trained on
# (ml/scripts/extract_sentinel2_patches.py). This is the single source of truth for
# the model's band contract; inference reads bands in file order, so a fetched stack
# MUST be in this order.
TRAINING_BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06",
                       "B07", "B08", "B8A", "B09", "B11", "B12"]

# S2 band id -> Earth Search v1 asset key
_ASSET_KEY = {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
              "B05": "rededge1", "B06": "rededge2", "B07": "rededge3", "B08": "nir",
              "B8A": "nir08", "B09": "nir09", "B11": "swir16", "B12": "swir22"}


def _flatten_lonlat(coords):
    """Yield [lon, lat] pairs from arbitrarily nested GeoJSON coordinate arrays."""
    if coords and isinstance(coords[0], (int, float)):
        yield coords
        return
    for c in coords:
        yield from _flatten_lonlat(c)


def geometry_bbox(geometry):
    """[min_lon, min_lat, max_lon, max_lat] from a GeoJSON geometry (WGS84 lon/lat)."""
    if not isinstance(geometry, dict) or "coordinates" not in geometry:
        raise ValueError("geometry must be a GeoJSON geometry object with 'coordinates'")
    pts = np.array(list(_flatten_lonlat(geometry["coordinates"])), dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("could not parse lon/lat pairs from geometry")
    return [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max())]


def _search_scene(bbox, start_date, end_date, max_cloud):
    body = {"collections": ["sentinel-2-l2a"], "bbox": bbox,
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z", "limit": 50}
    req = urllib.request.Request(STAC_SEARCH, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    feats = json.loads(urllib.request.urlopen(req, timeout=60).read())["features"]
    feats = [f for f in feats if f["properties"].get("eo:cloud_cover", 100) <= max_cloud]
    if not feats:
        raise RuntimeError(f"no Sentinel-2 L2A scene <= {max_cloud}% cloud for {bbox} "
                           f"in {start_date}..{end_date}")
    feats.sort(key=lambda f: f["properties"]["eo:cloud_cover"])
    return feats[0]


def _band_href(assets, band):
    key = _ASSET_KEY.get(band)
    if key in assets:
        return assets[key]["href"]
    for asset in assets.values():                       # fallback: match eo:bands name
        for eb in asset.get("eo:bands", []):
            if str(eb.get("name", "")).upper() == band.upper():
                return asset["href"]
    raise RuntimeError(f"asset for band {band} not found; available: {sorted(assets)}")


def fetch_sentinel2_stack(geometry, out_path,
                          start_date="2023-06-01", end_date="2023-09-30",
                          max_cloud=10):
    """Fetch a model-ready 12-band raw-DN Sentinel-2 GeoTIFF for a project boundary.

    Args:
        geometry: GeoJSON geometry (Polygon) in WGS84 lon/lat — a project boundary.
        out_path: where to write the GeoTIFF.
        start_date, end_date: acquisition window (YYYY-MM-DD); the least-cloudy scene
            in the window is used.
        max_cloud: reject scenes above this cloud-cover percentage.

    Returns:
        dict: {path, scene_id, cloud, date, bands, shape, mean_ndvi}.
        The output is directly consumable by CarbonCreditVerificationPipeline: bands in
        TRAINING_BAND_ORDER, raw DN, 10 m grid, cropped to the boundary bbox.
    """
    bbox = geometry_bbox(geometry)
    item = _search_scene(bbox, start_date, end_date, max_cloud)
    assets = item["assets"]

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_UNSAFESSL="YES"):
        # reference 10 m grid + scene UTM CRS from the red band
        with rasterio.open("/vsicurl/" + _band_href(assets, "B04")) as ref:
            crs = ref.crs
            utm_bounds = transform_bounds("EPSG:4326", crs, *bbox)
            win = from_bounds(*utm_bounds, transform=ref.transform)
            out_h = max(1, int(round(win.height)))
            out_w = max(1, int(round(win.width)))
            out_transform = window_transform(win, ref.transform)

        stack = []
        for band in TRAINING_BAND_ORDER:
            with rasterio.open("/vsicurl/" + _band_href(assets, band)) as src:
                w = from_bounds(*utm_bounds, transform=src.transform)
                arr = src.read(1, window=w, out_shape=(out_h, out_w),
                               resampling=Resampling.bilinear)
            stack.append(arr.astype("uint16"))
    stack = np.stack(stack)                              # [12, H, W] raw DN, training order

    profile = dict(driver="GTiff", height=out_h, width=out_w, count=12, dtype="uint16",
                   crs=crs, transform=out_transform, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)

    red, nir = stack[3].astype(float), stack[7].astype(float)   # B04, B08
    return {"path": out_path, "scene_id": item["id"],
            "cloud": item["properties"]["eo:cloud_cover"],
            "date": item["properties"]["datetime"][:10],
            "bands": TRAINING_BAND_ORDER, "shape": [12, out_h, out_w],
            "mean_ndvi": float(((nir - red) / (nir + red + 1e-6)).mean())}


if __name__ == "__main__":
    # Self-check: fetch a known-forest Amazon boundary (Hansen-verified in
    # docs/ml-forest-model-investigation.md) and assert the stack is MODEL-READY.
    # The checks are about data validity, not ecology — the module fetches whatever
    # the boundary covers (forest, water, cleared); NDVI is reported, not asserted.
    forest_aoi = {"type": "Polygon", "coordinates": [[
        [-59.67, -3.30], [-59.63, -3.30], [-59.63, -3.25], [-59.67, -3.25], [-59.67, -3.30]]]}
    meta = fetch_sentinel2_stack(forest_aoi, "acquisition_selfcheck.tif")
    with rasterio.open(meta["path"]) as s:
        assert s.count == 12, f"expected 12 bands, got {s.count}"
        a = s.read()
    assert a.shape[1] > 0 and a.shape[2] > 0, "empty raster"
    assert float((a == 0).all(axis=0).mean()) < 0.4, "stack is mostly empty (bad window)"
    assert float(a[7].std()) > 0, "no band variation — dead fetch"
    print(f"OK  scene={meta['scene_id']} cloud={meta['cloud']:.1f}% date={meta['date']} "
          f"shape={meta['shape']} NDVI={meta['mean_ndvi']:.3f} (location-dependent)")
