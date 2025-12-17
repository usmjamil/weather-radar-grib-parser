from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import xarray as xr
import cfgrib
import numpy as np
import json
import os
from typing import List, Dict, Any
from pydantic import BaseModel

app = FastAPI(title="GRIB2 Parser Service", version="1.0.0")

class RadarPoint(BaseModel):
    lat: float
    lon: float
    value: float

class ParsedData(BaseModel):
    points: List[RadarPoint]
    bounds: Dict[str, float]
    timestamp: str

@app.get("/")
async def root():
    return {"status": "ok", "service": "GRIB2 Parser"}

class ParseRequest(BaseModel):
    file_path: str

@app.post("/parse", response_model=ParsedData)
async def parse_grib2(request: ParseRequest):
    """
    Parse GRIB2 file and return radar data points
    """
    try:
        file_path = request.file_path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"GRIB2 file not found: {file_path}")
        
        # Open GRIB2 file with cfgrib
        ds = xr.open_dataset(file_path, engine='cfgrib')
        
        # Debug: print all available variables
        print(f"Available variables: {list(ds.data_vars.keys())}")
        print(f"Dimensions: {dict(ds.dims)}")
        
        # Find reflectivity data (common variable names)
        reflectivity_var = None
        possible_names = ['REFC', 'refc', 'Reflectivity', 'reflectivity', 'DBZ', 'dbz']
        
        for var_name in possible_names:
            if var_name in ds.data_vars:
                reflectivity_var = var_name
                break
        
        if reflectivity_var is None:
            # Use first available variable if reflectivity not found
            if ds.data_vars:
                reflectivity_var = list(ds.data_vars.keys())[0]
                print(f"Using variable: {reflectivity_var}")
            else:
                raise HTTPException(status_code=400, detail="No suitable data variable found in GRIB2 file")
        
        # Extract data
        data = ds[reflectivity_var].values
        
        # Handle coordinate extraction more robustly
        try:
            lat = ds.latitude.values
            lon = ds.longitude.values
        except AttributeError:
            # Try alternative coordinate names
            lat = ds.lat.values if 'lat' in ds else ds.y.values
            lon = ds.lon.values if 'lon' in ds else ds.x.values
        
        print(f"Data shape: {data.shape}, Lat shape: {lat.shape}, Lon shape: {lon.shape}")
        print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")
        print(f"Lat range: {np.min(lat)} to {np.max(lat)}")
        print(f"Lon range: {np.min(lon)} to {np.max(lon)}")
        
        # Create meshgrid for coordinates
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Flatten arrays and filter valid data
        # MRMS uses -999 for missing data
        valid_mask = ~np.isnan(data) & (data != -999) & (data >= -10) & (data <= 80)
        
        # Expand CONUS bounds to check if coordinates are in different format
        # MRMS might use 0-360 longitude instead of -180 to 180
        lat_min, lat_max = 25, 50
        lon_min, lon_max = -125, -65
        
        # Check if longitude is in 0-360 range and convert if needed
        if np.min(lon) >= 0:
            # Convert 0-360 to -180 to 180
            lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
            print("Converted longitude from 0-360 to -180-180 range")
        
        # Extract valid points within CONUS bounds
        valid_mask &= (lat_grid >= lat_min) & (lat_grid <= lat_max) & (lon_grid >= lon_min) & (lon_grid <= lon_max)
        
        print(f"Valid points found: {np.sum(valid_mask)}")
        
        points = []
        if np.sum(valid_mask) > 0:
            # Get indices of valid points
            valid_indices = np.where(valid_mask)
            
            for i in range(len(valid_indices[0])):
                row_idx = valid_indices[0][i]
                col_idx = valid_indices[1][i]
                
                points.append({
                    "lat": round(float(lat_grid[row_idx, col_idx]), 2),
                    "lon": round(float(lon_grid[row_idx, col_idx]), 2),
                    "value": round(float(data[row_idx, col_idx]), 1)
                })
        
        # Calculate bounds from valid points or use full grid
        if points:
            lats = [p["lat"] for p in points]
            lons = [p["lon"] for p in points]
            bounds = {
                "north": float(max(lats)),
                "south": float(min(lats)),
                "east": float(max(lons)),
                "west": float(min(lons))
            }
        else:
            # Use full grid bounds
            bounds = {
                "north": float(np.max(lat)),
                "south": float(np.min(lat)),
                "east": float(np.max(lon)),
                "west": float(np.min(lon))
            }
        
        # Get timestamp from file or use current time
        timestamp = ds.get('time', 'unknown').__str__() if 'time' in ds else "unknown"
        
        return ParsedData(
            points=points,
            bounds=bounds,
            timestamp=timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing GRIB2 file: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
