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

    # Load the blocks shapefile
    print("--- Loading blocks shapefile...")
    gdf = gpd.read_file(gdf_input_path)
    
    # Create the tessellations network
    print("--- Creating tessellations network...")
    nodes, edges = src.network_from_tessellation(gdf, crs, consolidate=consolidate)
    
    # Save the tessellations network locally
    print("--- Saving tessellations network...")
    nodes.to_file(gdf_output_path+"medellin_tessellations_nodes_before_consolidation.geojson")
    edges.to_file(gdf_output_path+"medellin_tessellations_edges_before_consolidation.geojson")

if __name__ == "__main__":

    # CRS (projected)
    crs = "EPSG:32618" # "EPSG:32618" for Medellín, "EPSG:32613" for Guadalajara
    # Consolidate the tessellations network?
    consolidate = (False,0)
    # Directory where city's blocks shapefile is stored
    gdf_input_path = "../data/input/shape/Medellin_blocks_DANE_2018_new/Medellin_blocks_DANE_2018_new.shp" 
    # Directory (folder) where the resulting tessellations network will be stored
    gdf_output_path = "../data/output/shape/network_tessellations/medellin/"
    main(gdf_input_path, crs, gdf_output_path)