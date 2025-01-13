import geopandas as gpd
import pandas as pd

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    import src
else:
    import src

def main(gdf_input_path, crs, gdf_output_path):
    # Load the GDL shapefile
    gdf = gpd.read_file(gdf_input_path)
    
    # Create a tessellation
    nodes, edges = src.network_from_tessellation(gdf, crs)
    
    # Save the tessellation
    nodes.to_file(gdf_output_path+"nodes_guadalajara_tessellation.geojson")
    edges.to_file(gdf_output_path+"edges_guadalajara_tessellation.geojson")

if __name__ == "__main__":
    crs = "EPSG:32613"
    gdf_input_path = "" # set input path: "data/blocks_guadalajara.shp"
    gdf_output_path = "" # set output path: "data/"
    main(gdf_input_path, crs, gdf_output_path)