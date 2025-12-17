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
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

app = FastAPI(title="GRIB2 Parser Service", version="1.0.0")

# Configuration
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '20'))  # More aggressive sampling for 512MB
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))   # Smaller chunks for memory
MAX_POINTS = int(os.getenv('MAX_POINTS', '5000'))  # Hard limit on points returned

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
    bounds: Optional[Dict[str, float]] = None  # Optional geographic bounds

@app.post("/parse", response_model=ParsedData)
async def parse_grib2(request: ParseRequest):
    """
    Download and parse GRIB2 file from URL and return radar data points
    Memory optimized for 512MB servers
    """
    try:
        url = request.url
        bounds = request.bounds
        
        print(f"Starting memory-optimized GRIB2 parsing...")
        print(f"Sample rate: {SAMPLE_RATE}, Chunk size: {CHUNK_SIZE}, Max points: {MAX_POINTS}")
        
        with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as temp_file:
            file_path = temp_file.name
            
            print(f"Downloading MRMS data from {url}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            if url.endswith('.gz'):
                print("Decompressing gzip file...")
                decompressed_data = gzip.decompress(response.content)
                temp_file.write(decompressed_data)
            else:
                temp_file.write(response.content)
            temp_file.flush()
        
        try:
            print("Opening GRIB2 file with cfgrib...")
            ds = xr.open_dataset(file_path, engine='cfgrib')
            
            # Get the reflectivity variable
            var_name = None
            for var in ds.data_vars:
                if 'refl' in var.lower() or 'reflectivity' in var.lower():
                    var_name = var
                    break
            
            if not var_name:
                # Try to find any numeric variable
                for var in ds.data_vars:
                    if ds[var].dtype.kind in ['f', 'i']:  # float or integer
                        var_name = var
                        break
            
            if not var_name:
                raise HTTPException(status_code=500, detail="No suitable data variable found")
            
            data = ds[var_name].values
            lat = ds['latitude'].values
            lon = ds['longitude'].values
            
            print(f"Data shape: {data.shape}, Memory usage: {data.nbytes / 1024 / 1024:.1f} MB")
            
            # Apply geographic bounds if provided
            lat_min, lat_max, lon_min, lon_max = -90, 90, -180, 180
            if bounds:
                lat_min = bounds.get('south', -90)
                lat_max = bounds.get('north', 90)
                lon_min = bounds.get('west', -180)
                lon_max = bounds.get('east', 180)
                print(f"Using bounds: {lat_min} to {lat_max}, {lon_min} to {lon_max}")
            
            # Convert longitude from 0-360 to -180-180 if needed
            if np.max(lon) > 180:
                lon = np.where(lon > 180, lon - 360, lon)
            
            # Create valid mask with bounds
            valid_mask = (
                (data > 0) &  # Only positive reflectivity values
                (~np.isnan(data)) &  # Remove NaN values
                (lat >= lat_min) & (lat <= lat_max) & 
                (lon >= lon_min) & (lon <= lon_max)
            )
            
            total_valid = np.sum(valid_mask)
            print(f"Valid points found: {total_valid}")
            
            points = []
            if total_valid > 0:
                # Get indices of valid points
                valid_indices = np.where(valid_mask)
                
                # Calculate effective sampling rate to stay within MAX_POINTS
                effective_sample_rate = max(1, total_valid // MAX_POINTS)
                final_sample_rate = max(SAMPLE_RATE, effective_sample_rate)
                
                print(f"Using sample rate: {final_sample_rate} (effective: {effective_sample_rate})")
                
                # Process in very small chunks for memory efficiency
                chunk_count = 0
                for chunk_start in range(0, total_valid, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_valid)
                    
                    # Sample within this chunk
                    for i in range(chunk_start, min(chunk_end, total_valid), final_sample_rate):
                        if len(points) >= MAX_POINTS:
                            break
                            
                        row_idx = valid_indices[0][i]
                        col_idx = valid_indices[1][i]
                        
                        lat_val = float(lat[row_idx, col_idx])
                        lon_val = float(lon[row_idx, col_idx])
                        value_val = float(data[row_idx, col_idx])
                        
                        points.append({
                            "lat": lat_val,
                            "lon": lon_val,
                            "value": value_val
                        })
                    
                    chunk_count += 1
                    
                    # Memory cleanup
                    if chunk_count % 5 == 0:
                        import gc
                        gc.collect()
                        print(f"Processed chunk {chunk_count}, points so far: {len(points)}")
                    
                    if len(points) >= MAX_POINTS:
                        break
                
                print(f"Final result: {len(points)} points from {total_valid} total points")
            
            # Calculate bounds from actual data
            if points:
                lats = [p["lat"] for p in points]
                lons = [p["lon"] for p in points]
                data_bounds = {
                    "north": max(lats),
                    "south": min(lats),
                    "east": max(lons),
                    "west": min(lons)
                }
            else:
                data_bounds = {"north": 0, "south": 0, "east": 0, "west": 0}
            
            # Get timestamp from dataset
            timestamp = str(ds.time.values[0]) if 'time' in ds else "unknown"
            
            # Cleanup dataset to free memory
            ds.close()
            import gc
            gc.collect()
            
            return ParsedData(points=points, bounds=data_bounds, timestamp=timestamp)
            
        except Exception as e:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.unlink(file_path)
            raise HTTPException(status_code=500, detail=f"Error parsing GRIB2 file: {str(e)}")
        finally:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.unlink(file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
