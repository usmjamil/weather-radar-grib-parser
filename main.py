from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import xarray as xr
import cfgrib
import numpy as np
import json
import os
import requests
import tempfile
import gzip
from typing import List, Dict, Any
from pydantic import BaseModel

app = FastAPI(title="GRIB2 Parser Service", version="1.0.0")

SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '10')) 
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))

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
    url: str

@app.post("/parse", response_model=ParsedData)
async def parse_grib2(request: ParseRequest):
    """
    Download and parse GRIB2 file from URL and return radar data points
    """
    try:
        url = request.url
        
        with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as temp_file:
            file_path = temp_file.name
            
            
            print(f"Downloading MRMS data from {url}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            
            if url.endswith('.gz'):
                decompressed_data = gzip.decompress(response.content)
                temp_file.write(decompressed_data)
            else:
                temp_file.write(response.content)
            temp_file.flush()
            
            print(f"Downloaded and saved to {file_path}")
        
        try:
            ds = xr.open_dataset(file_path, engine='cfgrib')
        
            print(f"Available variables: {list(ds.data_vars.keys())}")
            print(f"Dimensions: {dict(ds.dims)}")
            
            reflectivity_var = None
            possible_names = ['REFC', 'refc', 'Reflectivity', 'reflectivity', 'DBZ', 'dbz']
            
            for var_name in possible_names:
                if var_name in ds.data_vars:
                    reflectivity_var = var_name
                    break
            
            if reflectivity_var is None:
                if ds.data_vars:
                    reflectivity_var = list(ds.data_vars.keys())[0]
                    print(f"Using variable: {reflectivity_var}")
                else:
                    raise HTTPException(status_code=400, detail="No data variables found in GRIB2 file")
            
            print(f"Using variable: {reflectivity_var}")
            
            data = ds[reflectivity_var].values
            
            try:
                lat = ds.latitude.values
                lon = ds.longitude.values
            except AttributeError:
                lat = ds.lat.values if 'lat' in ds else ds.y.values
                lon = ds.lon.values if 'lon' in ds else ds.x.values
            
            print(f"Data shape: {data.shape}, Lat shape: {lat.shape}, Lon shape: {lon.shape}")
            print(f"Data range: {np.nanmin(data)} to {np.nanmax(data)}")
            print(f"Lat range: {np.min(lat)} to {np.max(lat)}")
            print(f"Lon range: {np.min(lon)} to {np.max(lon)}")
            
        
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            
            
            
            valid_mask = ~np.isnan(data) & (data != -999) & (data >= -10) & (data <= 80)
            
            
            
            lat_min, lat_max = 25, 50
            lon_min, lon_max = -125, -65
            
            
            if np.min(lon) >= 0:
                
                lon_grid = np.where(lon_grid > 180, lon_grid - 360, lon_grid)
                print("Converted longitude from 0-360 to -180-180 range")
            
            
            valid_mask &= (lat_grid >= lat_min) & (lat_grid <= lat_max) & (lon_grid >= lon_min) & (lon_grid <= lon_max)
            
            print(f"Valid points found: {np.sum(valid_mask)}")
            
            points = []
            if np.sum(valid_mask) > 0:
            
                valid_indices = np.where(valid_mask)
                total_points = len(valid_indices[0])
                
                print(f"Processing {total_points} points in chunks of {CHUNK_SIZE}")
                
                chunk_count = 0
                for chunk_start in range(0, total_points, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_points)
                    chunk_indices = range(chunk_start, chunk_end, SAMPLE_RATE)
                    
                    chunk_points = []
                    for i in chunk_indices:
                        row_idx = valid_indices[0][i]
                        col_idx = valid_indices[1][i]
                        
                        lat_val = float(lat_grid[row_idx, col_idx])
                        lon_val = float(lon_grid[row_idx, col_idx])
                        value_val = float(data[row_idx, col_idx])
                        
                        chunk_points.append({
                            "lat": lat_val,
                            "lon": lon_val,
                            "value": value_val
                        })
                    
                    points.extend(chunk_points)
                    chunk_count += 1
                    
                    if chunk_count % 10 == 0:  # Log every 10 chunks
                        print(f"Processed {chunk_count} chunks, {len(points)} points so far")
                
                print(f"Final result: {len(points)} points from {total_points} total points (1 in {SAMPLE_RATE})")
            
            
            if points:
                lats = [p["lat"] for p in points]
                lons = [p["lon"] for p in points]
                bounds = {
                    "north": max(lats),
                    "south": min(lats),
                    "east": max(lons),
                    "west": min(lons)
                }
            else:
                
                bounds = {
                    "north": 50,
                    "south": 25,
                    "east": -65,
                    "west": -125
                }
            
            
            timestamp = ds.get('time', 'unknown').__str__() if 'time' in ds else "unknown"
            
            return ParsedData(
                points=points,
                bounds=bounds,
                timestamp=timestamp
            )
        
        except Exception as e:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.unlink(file_path)
            raise HTTPException(status_code=500, detail=f"Error parsing GRIB2 file: {str(e)}")
    
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.unlink(file_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
