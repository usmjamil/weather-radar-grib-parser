from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xarray as xr
import numpy as np
import requests
import tempfile
import gzip
import os
from typing import List, Dict, Optional

app = FastAPI(title="GRIB2 Parser Service", version="1.0.0")

MAX_POINTS = int(os.getenv("MAX_POINTS", "3000"))
class RadarPoint(BaseModel):
    lat: float
    lon: float
    value: float
class ParsedData(BaseModel):
    points: List[RadarPoint]
    bounds: Dict[str, float]
    timestamp: str
class ParseRequest(BaseModel):
    url: str
    bounds: Optional[Dict[str, float]] = None

def download_grib(url: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        if url.endswith(".gz"):
            f.write(gzip.decompress(resp.content))
        else:
            f.write(resp.content)
        return f.name

@app.post("/parse", response_model=ParsedData)
async def parse_grib2(request: ParseRequest):
    try:
        file_path = download_grib(request.url)

        ds = xr.open_dataset(
            file_path,
            engine="cfgrib"
        )

        print(f"Dataset variables: {list(ds.data_vars)}")
        print(f"Dataset coordinates: {list(ds.coords)}")
        print(f"Dataset dims: {dict(ds.dims)}")

        var = None
        for v in ds.data_vars:
            if ds[v].dtype.kind in ("f", "i"):
                var = v
                print(f"Selected variable: {v}, shape: {ds[v].shape}, dtype: {ds[v].dtype}")
                break
        
        if not var:
            raise HTTPException(500, "No numeric variable found")

        if "time" in ds[var].dims:
            field = ds[var].isel(time=0)
        else:
            field = ds[var]
        
        print(f"Field shape before loading: {field.shape}")
        field = field.load()
        print(f"Field shape after loading: {field.shape}")

        if "latitude" in ds.coords and "longitude" in ds.coords:
            lat = ds["latitude"].values
            lon = ds["longitude"].values
        elif "lat" in ds.coords and "lon" in ds.coords:
            lat = ds["lat"].values
            lon = ds["lon"].values
        else:
            lat_var = next((v for v in ds.data_vars if "lat" in v.lower()), None)
            lon_var = next((v for v in ds.data_vars if "lon" in v.lower()), None)
            if lat_var and lon_var:
                lat = ds[lat_var].values
                lon = ds[lon_var].values
            else:
                raise HTTPException(500, "Could not find latitude/longitude coordinates")
        
        print(f"Lat shape: {lat.shape}, Lon shape: {lon.shape}")
        print(f"Data shape: {field.shape}")

        data = field.values

        lon2d, lat2d = np.meshgrid(lon, lat)
        print(f"Created 2D meshes - Lat2D: {lat2d.shape}, Lon2D: {lon2d.shape}")

        if lon2d.max() > 180:
            lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)

        b = request.bounds or {}
        lat_min, lat_max = b.get("south", -90), b.get("north", 90)
        lon_min, lon_max = b.get("west", -180), b.get("east", 180)

        mask = (
            (data > 0)
            & ~np.isnan(data)
            & (lat2d >= lat_min)
            & (lat2d <= lat_max)
            & (lon2d >= lon_min)
            & (lon2d <= lon_max)
        )
        
        print(f"Mask shape: {mask.shape}, Valid points: {np.sum(mask)}")

        idx = np.column_stack(np.where(mask))
        print(f"Indices shape: {idx.shape}")
        
        if len(idx) == 0:
            print("No valid points found")
            return ParsedData(points=[], bounds={}, timestamp="unknown")

        step = max(1, len(idx) // MAX_POINTS)

        points: List[RadarPoint] = []
        print(f"Starting to process {len(idx)} points with step {step}")
        
        for i, (r, c) in enumerate(idx[::step]):
            if len(points) >= MAX_POINTS:
                break
            if i < 5: 
                print(f"Point {i}: r={r}, c={c}, lat={lat2d[r, c]}, lon={lon2d[r, c]}, val={data[r, c]}")
            points.append(
                RadarPoint(
                    lat=float(lat2d[r, c]),
                    lon=float(lon2d[r, c]),
                    value=float(data[r, c]),
                )
            )

        print(f"Successfully created {len(points)} points")

        bounds = {
            "north": max(p.lat for p in points) if points else 0,
            "south": min(p.lat for p in points) if points else 0,
            "east": max(p.lon for p in points) if points else 0,
            "west": min(p.lon for p in points) if points else 0,
        }

        timestamp = "unknown"
        try:
            if "time" in ds.coords:
                time_val = ds.time.values[0]
                timestamp = str(time_val)
            else:
                timestamp = "unknown"
        except Exception as time_error:
            print(f"Error getting timestamp: {time_error}")
            timestamp = "unknown"

        print(f"Calculated bounds: {bounds}")
        print(f"Timestamp: {timestamp}")

        ds.close()
        os.unlink(file_path)

        print(f"Returning {len(points)} points")
        return ParsedData(
            points=points,
            bounds=bounds,
            timestamp=timestamp,
        )

    except Exception as e:
        raise HTTPException(500, str(e))
@app.get("/health")
def health():
    return {"status": "ok"}
