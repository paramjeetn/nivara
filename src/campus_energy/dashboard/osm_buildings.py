import json
import random
from pathlib import Path

def generate_procedural_buildings(output_file: Path):
    if output_file.exists():
        return
        
    features = []
    # Stanford bounding box roughly: 37.4220, -122.1770 to 37.4330, -122.1620
    min_lat, min_lon = 37.4220, -122.1770
    max_lat, max_lon = 37.4330, -122.1620
    
    # Generate 100 random rectangular buildings
    for i in range(100):
        # Random center
        clat = random.uniform(min_lat + 0.001, max_lat - 0.001)
        clon = random.uniform(min_lon + 0.001, max_lon - 0.001)
        
        # Random size (in degrees roughly 10-30 meters)
        w = random.uniform(0.0001, 0.0004)
        h = random.uniform(0.0001, 0.0004)
        
        height = random.uniform(10, 35) # meters
        
        polygon = [
            [clon - w, clat - h],
            [clon + w, clat - h],
            [clon + w, clat + h],
            [clon - w, clat + h],
            [clon - w, clat - h]
        ]
        
        features.append({
            "type": "Feature",
            "properties": {"height": height},
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            }
        })
        
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)

if __name__ == "__main__":
    generate_procedural_buildings(Path("data/generated/stanford_buildings.geojson"))
