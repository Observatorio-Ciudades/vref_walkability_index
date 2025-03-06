import geopandas as gpd
import osmnx as ox
import igraph as ig
from rtree import index
import shapely as sh
from shapely.geometry import LineString, Point, Polygon
import momepy
import numpy as np
from scipy.spatial import Voronoi
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import pandas as pd


def get_boeing_network(zone_blocks,nodes,edges,buff):
    '''

    '''

    # Usar unary_union para fusionar todos los polígonos en una geometría única
    poligono_exterior = zone_blocks.union_all().convex_hull.buffer(buff)

    # Crear un nuevo GeoDataFrame con el polígono exterior
    zona = gpd.GeoDataFrame(geometry=gpd.GeoSeries([poligono_exterior]), crs=zone_blocks.crs)
    zona = zona.to_crs('epsg:4326')

    # Filtrar los edges de boeing de la zona
    zone_boeing_edges = gpd.sjoin(edges, zona, predicate='intersects')
    zone_boeing_edges = zone_boeing_edges.drop(['index_right'], axis=1)

    # Obtener los nodos que corresponden a los edges filtrados
    filtered_node_ids = set(zone_boeing_edges['from']).union(set(zone_boeing_edges['to']))

    # Filtrar los nodos de boeing de la zona
    zone_boeing_nodes = nodes[nodes['ID'].isin(filtered_node_ids)]

    return zone_boeing_nodes,zone_boeing_edges


def network_entities(nodes:gpd.GeoDataFrame, edges:gpd.GeoDataFrame, crs='epsg:32618'):
	"""
	Create a network based on nodes and edges without unique ids and to - from attributes.

	Parameters:
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes for network in EPSG:4326
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges for network in EPSG:4326

	Returns:
		geopandas.GeoDataFrame: nodes GeoDataFrame with unique ids based on coordinates named osmid in crs
		geopandas.GeoDataFrame: edges GeoDataFrame with to - from attributes based on nodes ids named u and v respectively in crs
	"""

	# Copy to avoid editing original GeoDataFrames
	nodes = nodes.copy()
	edges = edges.copy()
	# Change coordinate system to meters for unique ids
	nodes = nodes.to_crs(crs)
	edges = edges.to_crs(crs)
	# Unique str id for nodes based on coordinates
	nodes['osmid'] = (nodes['geometry'].x).astype(str)+(nodes['geometry'].y).astype(str)
	# Create columns [u] and [v] in edges for stablishing to and from
	edges['u'] = ''
	edges['v'] = ''
	# Extract start and end coordinates for [u,v] columns
	for index, row in edges.iterrows():
		edges.at[index,'u'] = str(list(row['geometry'].coords)[0][0])+str(list(row['geometry'].coords)[0][1])
		edges.at[index,'v'] = str(list(row['geometry'].coords)[-1][0])+str(list(row['geometry'].coords)[-1][1])
	# Add key column for compatibility with osmnx
	edges['key'] = 0
	# Calculate edges lentgh
	edges['length'] = edges.to_crs(crs).length
    # Organice nodes and edges
	nodes = nodes.set_index('osmid')
	nodes['x'] = nodes['geometry'].x
	nodes['y'] = nodes['geometry'].y
	edges = edges.set_index(['u','v','key'])

	return nodes, edges


def network_from_tessellation_ig_rtree(blocks:gpd.GeoDataFrame):
    '''
    Function to create a network based on the tessellations conformed from a set of polygons

    Args:
        blocks (gpd.GeoDataFrame): Blocks uniquely indexed from which the tessellations will be performed
    Returns:
        nodes_consolidated (gpd.GeoDataFrame): Nodes of the network representing the intersections
        edges_consolidated (gpd.GeoDataFrame): Edges of the network representing conections
        tessellation_gdf (gpd.GeoDataFrame): Morphological tessellations build from the polygons
    '''

    ## Preliminars

    # Copy to avoid editing original GeoDataFrames
    buildings = blocks.copy()
    id_name = buildings.index.name
    buildings = blocks.reset_index()
    # Create dictionary mapping IDs to indexes
    index_to_id = {index_: id_ for index_, id_ in enumerate(buildings[id_name])}
    print('Analyzing touching polygons...\n')
    # Create spatial index
    spatial_index = index.Index()
    for pos, poly in enumerate(buildings.geometry):
        spatial_index.insert(pos, poly.bounds)
    # Create an empty graph
    G = ig.Graph()
    G.add_vertices(len(buildings))
    # Add edges if polygons touch using the spatial index
    edges = []
    for i, poly1 in enumerate(buildings.geometry):
        possible_matches_index = list(spatial_index.intersection(poly1.bounds))
        for j in possible_matches_index:
            if i != j and poly1.touches(buildings.geometry.iloc[j]):
                edges.append((i, j))
    G.add_edges(edges)
    # Find connected components
    clusters = G.connected_components()
    # Combine the polygons within each component
    partial_polygons = []
    partial_polygons_index = []
    for cluster in clusters:
        combined_poly = sh.ops.unary_union([buildings.geometry.iloc[idx] for idx in cluster])
        partial_polygons_index.append(cluster[0])
        if combined_poly.geom_type == 'Polygon':
            partial_polygons.append(combined_poly)
        elif combined_poly.geom_type == 'MultiPolygon':
            partial_polygons.append(sh.ops.unary_union([poly.buffer(0.05) for poly in combined_poly.geoms]))
    # Create a new GeoDataFrame with the combined polygons
    partial_unified_gdf = gpd.GeoDataFrame(index=partial_polygons_index,geometry=partial_polygons, crs=buildings.crs)
    partial_new_buildings = partial_unified_gdf.reset_index().copy()
    partial_new_buildings[id_name] = partial_new_buildings['index'].replace(index_to_id)
    # Create a negative buffer to avoid remaining connected polygons
    partial_new_buildings['geometry'] = partial_new_buildings.buffer(-1)
    # Simplify geometry
    partial_new_buildings['geometry'] = partial_new_buildings['geometry'].simplify(tolerance=1, preserve_topology=True)
    # Correct the polygons
    polygons = []
    polygons_index = []
    for index_, poly in zip(partial_new_buildings[id_name],partial_new_buildings.geometry):
        if poly.geom_type == 'Polygon':
            polygons.append(poly)
            polygons_index.append(index_)
        elif poly.geom_type == 'MultiPolygon':
            for i,poly1 in enumerate(poly.geoms):
                polygons.append(poly1)
                polygons_index.append(f'{index_}_{i}')
    # Create a new GeoDataFrame with the combined polygons
    unified_gdf = gpd.GeoDataFrame(index=polygons_index,geometry=polygons, crs=partial_new_buildings.crs)
    unified_gdf[id_name] = unified_gdf.index
    new_buildings = unified_gdf.reset_index(drop=True).copy()
    
    ## TESELLATIONS

    # Calculate morphological tessellations
    limit = momepy.buffered_limit(new_buildings, buffer=50)
    tessellation_gdf = momepy.Tessellation(new_buildings, unique_id=id_name, limit=limit, shrink=0.1, segment=0.1).tessellation

    ## NETWORK

    print('\nConforming network...\n')
    # Simplify the tesellations
    tessellation_gdf['geometry'] = tessellation_gdf['geometry'].simplify(tolerance=0.2, preserve_topology=True)
    # Extract the lines and points conforming the polygons
    points_list = []
    lines_list = []
    for idx, tessellation in tessellation_gdf.iterrows():
        exterior_coords = tessellation.geometry.exterior.coords
        for i in range(len(exterior_coords) - 1):
            point = Point(exterior_coords[i])
            points_list.append(point)
            line = LineString([exterior_coords[i], exterior_coords[i + 1]])
            lines_list.append(line)
    points_gdf = gpd.GeoDataFrame(geometry=points_list,crs=tessellation_gdf.crs).drop_duplicates(['geometry'])
    lines_gdf = gpd.GeoDataFrame(geometry=lines_list,crs=tessellation_gdf.crs).drop_duplicates()
    # Organize the points and lines to a proper network structure
    nodes, edges = network_entities(points_gdf, lines_gdf, crs=buildings.crs)
    # Create a graph entity with osmnx
    G = ox.graph_from_gdfs(nodes, edges)
    # Simplifies and consolidate the network
    G_simplified = ox.simplification.simplify_graph(G)
    G_consolidated = ox.simplification.consolidate_intersections(G_simplified, tolerance=5)
    nodes_consolidated, edges_consolidated = ox.graph_to_gdfs(G_consolidated)
    # Organize the nodes
    nodes_consolidated = nodes_consolidated[['geometry']]
    nodes_consolidated = nodes_consolidated.reset_index()
    nodes_consolidated = nodes_consolidated.set_crs(buildings.crs,allow_override=True)
    # Organize the edges
    edges_consolidated = edges_consolidated[['geometry']]
    edges_consolidated = edges_consolidated.reset_index()
    edges_consolidated['id'] = edges_consolidated.index
    edges_consolidated = edges_consolidated.set_crs(buildings.crs, allow_override=True)

    # Elminate the duplicate edges
    def create_edge_key(row):
        # Function to create a key that does not distinguish the direction of the edge
        return tuple(sorted([row['u'], row['v']]))
    
    # Create a new 'edge_key' column to identify unique edges
    edges_consolidated['edge_key'] = edges_consolidated.apply(create_edge_key, axis=1)
    # Remove duplicates based on 'edge_key'
    edges_consolidated = edges_consolidated.drop_duplicates(subset='edge_key')
    # Remove the 'edge_key' column as it is not necessary in the final result
    edges_consolidated = edges_consolidated.drop(columns=['edge_key'])
    # Re simplify the network
    nodes_consolidated['x'] = nodes_consolidated['geometry'].x
    nodes_consolidated['y'] = nodes_consolidated['geometry'].y
    nodes_consolidated = nodes_consolidated.set_index(['osmid'])
    edges_consolidated = edges_consolidated.set_index(['u','v','key'])
    G = ox.graph_from_gdfs(nodes_consolidated, edges_consolidated)
    G_simplified = ox.simplification.simplify_graph(G)
    nodes_consolidated, edges_consolidated = ox.graph_to_gdfs(G_simplified)
    # Organize the nodes
    nodes_consolidated = nodes_consolidated[['geometry']]
    nodes_consolidated = nodes_consolidated.reset_index()
    nodes_consolidated = nodes_consolidated.set_crs(buildings.crs,allow_override=True)
    # Organize the edges
    edges_consolidated = edges_consolidated[['geometry']]
    edges_consolidated = edges_consolidated.reset_index()
    edges_consolidated['id'] = edges_consolidated.index
    edges_consolidated = edges_consolidated.set_crs(buildings.crs, allow_override=True)

    return nodes_consolidated, edges_consolidated, tessellation_gdf


def elevation_DEM(nodes:gpd.GeoDataFrame, edges:gpd.GeoDataFrame, DEM_path):
    '''
    Function to add elevation to a network based on a DEM

    Args:
        nodes (gpd.GeoDataFrame): Nodes from the network uniquely indexed by osmid
        edges (gpd.GeoDataFrame): Edges from the network uniquely indexed by u, v, key
        DEM_path: Path where the DEM is stored, it must be in espg:4326 (WSG84)
    Returns:
        nodes (gpd.GeoDataFrame): Network's nodes with raster elevation
        edges (gpd.GeoDataFrame): Network's edges with lenght, grade and grade_abs
    '''

    # Copy to avoid editing original GeoDataFrames
    nodes = nodes.copy()
    edges = edges.copy()
    # Organice nodes and edges
    nodes = nodes.to_crs('epsg:4326')
    edges = edges.to_crs('epsg:4326')
    nodes['x'] = nodes['geometry'].x
    nodes['y'] = nodes['geometry'].y
    # Graph
    G = ox.graph_from_gdfs(nodes, edges)
    # Elevations based on DEM
    G = ox.elevation.add_node_elevations_raster(G, DEM_path)
    # Add edge's lenght
    G = ox.distance.add_edge_lengths(G)
    # Add edge's grade
    G = ox.elevation.add_edge_grades(G)
    nodes, edges = ox.graph_to_gdfs(G)
    
    return nodes, edges


def voronoi_polygons(nodes:gpd.GeoDataFrame):
    '''
    Function to create Voronoi polygons from a set of nodes

    Args:
        nodes (gpd.GeoDataFrame): Nodes stored in a GeoDataFrame indexed by an unique id column

    Returns:
        voronoi_gdf (gpd.GeoDataFrame): Voronoi polygons
    '''

    # Copy to avoid editing original GeoDataFrames
    nodes = nodes.copy()
    # Extract the coordinates of the points
    points = np.array(nodes.geometry.apply(lambda p: (p.x, p.y)).tolist())
    # Generate Voronoi polygons
    vor = Voronoi(points)
    polygons = []
    for region in vor.regions:
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            polygons.append(Polygon(polygon))
    voronoi_gdf = gpd.GeoDataFrame({'geometry': polygons},crs=nodes.crs)
    # Create the polygon that encloses all the points
    poligono_exterior = nodes.union_all().convex_hull
    # Filter polygons according to the analysis area
    voronoi_gdf = voronoi_gdf[voronoi_gdf.geometry.within(poligono_exterior)]
    # Set the intersection ID to each Voronoi polygon
    # voronoi_gdf = gpd.sjoin(voronoi_gdf,nodes[[id,'geometry']],how='left',predicate='intersects').drop(['index_right'],axis=1)
    voronoi_gdf = gpd.sjoin(voronoi_gdf, nodes[['geometry']], how='left', predicate='intersects')
    # Index the voronoi geodataframe by the same index of the entry nodes
    voronoi_gdf = voronoi_gdf.set_index(nodes.index.name)

    return voronoi_gdf


def assing_blocks_attribute_to_voronoi(blocks:gpd.GeoDataFrame, voronoi:gpd.GeoDataFrame, attribute_column:str):
    '''
    Function to assing a block's attribute to Voronoi polygons based on the intersection of areas

    Args:
        blocks (gpd.GeoDataFrame): Blocks stored in a GeoDataFrame indexed by an unique id column
        voronoi (gpd.GeoDataFrame): Voronoi polygons stored in a GeoDataFrame indexed by an unique id column
        attribute_column (str): Name of the column where the attribute is stored

    Returns:
        voronoi (gpd.GeoDataFrame): Voronoi polygons with the corresponding proportion of the desired attribute
    '''

    # Copy to avoid editing original GeoDataFrames
    blocks = blocks.copy()
    voronoi = voronoi.copy()

    index_voronoi = voronoi.index.name
    index_blocks = blocks.index.name
    blocks = blocks.reset_index()
    voronoi = voronoi.reset_index()

    # Perform the intersection of geodataframes
    intersections = gpd.overlay(blocks[[index_blocks,'geometry']], voronoi, how='intersection')
    # Calculate the area of ​​intersections
    intersections['area_intersection'] = intersections.geometry.area
    # Calculate the area of ​​the original blocks
    blocks['area_block'] = blocks.geometry.area
    # Associate the original area of ​​the blocks to the intersections
    intersections = intersections.merge(blocks[[index_blocks,'area_block', attribute_column]], on=index_blocks)
    # Calculate the proportion of population at intersections
    intersections['column_intersection'] = intersections[attribute_column] * (intersections['area_intersection'] / intersections['area_block'])
    intersections['column_intersection'] = np.round(intersections['column_intersection'],0).astype(int)
    # Group by zones to obtain the total population in each zone
    population_zones = intersections.dissolve(by=index_voronoi, aggfunc='sum')[['column_intersection']]
    # Rename the column for clarity
    population_zones = population_zones.rename(columns={'column_intersection': attribute_column})
    # Add the calculation to the original zones
    voronoi = voronoi.merge(population_zones, on=index_voronoi, how='left').fillna(0.0)
    voronoi[attribute_column] = voronoi[attribute_column].astype(int)

    return voronoi

# Define the kernels
def epanechnikov_kernel(dist, bandwidth):
    return 0.75 * (1 - (dist / bandwidth) ** 2) if dist < bandwidth else 0
def quartic_kernel(dist, bandwidth):
    return (15 / 16) * ((1 - (dist / bandwidth) ** 2) ** 2) if dist < bandwidth else 0


# Function to calculate density
def calculate_density(points, bandwidth, pixel_size, kernel_shape):
    '''
    Calculate a density map for a set of points using kernel density estimation (KDE).

    This function computes the density of points within a geographic area based on a selected
    kernel function, bandwidth, and grid resolution. It returns the updated points with
    assigned densities, the density grid, and the area boundaries.

    Args:
    points (gpd.GeoDataFrame): A GeoDataFrame containing the points for density calculation. Must 
        include a 'geometry' column with Point geometries.
    bandwidth (float): The radius of influence for the kernel function. Determines the area of effect for each point.
    pixel_size (float): The size of each grid cell in the output density map. Defines the resolution of the density grid.
    kernel_shape (str): The type of kernel function to use for density calculation. Options are:
        - 'quartic': Quartic (biweight) kernel
        - 'epanechnikov': Epanechnikov kernel

    Returns:
    points (gpd.GeoDataFrame): The input GeoDataFrame with an additional column 'density', indicating 
        the density value for each point.
    density (np.ndarray): A 2D NumPy array representing the density grid, with rows and columns corresponding 
        to the y and x coordinates of the grid.
    x_min (float): The minimum x-coordinate of the area boundary.
    y_min (float): The minimum y-coordinate of the area boundary.
    x_max (float): The maximum x-coordinate of the area boundary.
    y_max (float): The maximum y-coordinate of the area boundary.
    '''

    # Copy to avoid editing original GeoDataFrames
    points = points.copy()
    # Select the kernel
    kernel_list = {'quartic': quartic_kernel, 'epanechnikov': epanechnikov_kernel}
    if kernel_shape not in kernel_list.keys():
        raise KeyError(f'Invalid kernel. Available kernels are {[i for i in kernel_list.keys()]}')
    kernel_function = kernel_list[kernel_shape]

    # Get the limits of the area of the points
    x_min, y_min, x_max, y_max = points.total_bounds

    # Create grid of points for calculation
    x_grid = np.arange(x_min, x_max, pixel_size)
    y_grid = np.arange(y_min, y_max, pixel_size)
    density = np.zeros((y_grid.size, x_grid.size))

    # Create a STRtree for quick neighbor search
    tree = STRtree(points.geometry)

    # Calculate the density in each grid cell
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            cell_center = Point(x, y)
            # Obtener índices de los vecinos
            neighbor_indices = tree.query(cell_center.buffer(bandwidth))
            density_value = 0
            for idx in neighbor_indices:
                neighbor_geom = points.geometry.iloc[idx]  # Get the geometry of the index
                dist = cell_center.distance(neighbor_geom)
                density_value += kernel_function(dist, bandwidth)
            density[j, i] = density_value

    points['density'] = 0.0

    # Create a GeoDataFrame for the mesh
    mesh_points = []
    density_values = []
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            mesh_points.append(Point(x, y))
            density_values.append(density[j, i])

    mesh_gdf = gpd.GeoDataFrame({'geometry': mesh_points, 'density': density_values}, crs=points.crs)

    # Assign density values to the original GeoDataFrame
    for i, point in points.iterrows():
        nearest_cell = mesh_gdf.geometry.distance(point.geometry).idxmin()
        points.at[i, 'density'] = mesh_gdf.at[nearest_cell, 'density']

    # Return points and density, plus limits for use outside the function
    return points, density, x_min, y_min, x_max, y_max


# Function to plot density
def plot_density(points, density):
    # Create the figure y and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Show the density matrix as an image
    x_min, y_min, x_max, y_max = points.total_bounds
    cax = ax.imshow(density, cmap='hot', extent=[x_min, x_max, y_min, y_max], origin='lower')
    
    # Add a color bar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.6)  # Ajusta el valor de shrink según sea necesario
    cbar.set_label('Density')
    
    # Add titles and tags
    ax.set_title('Kernel Density Estimation')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Show the graph
    plt.show()


def calcular_entropia(osmid, n_land_use, porcentaje_usos_edges, categories):
    '''
    Calculates entropy for a specific element identified by its osmid,
    considering the number of land use categories and the area proportions
    associated with those categories.

    Args:
        osmid (int or str): Unique identifier of the element for which entropy is calculated.
        n_land_use (pd.Series): Series indexed by osmid containing the number of land use categories 
            (n in the formula) for each element.
        porcentaje_usos_edges (pd.Series): MultiIndex Series indexed by (osmid, land_use) containing 
            the area proportions (P_k) of each land use category associated with an element.

    Returns:
        entropia (float): Calculated entropy value:
            - Returns 1 if there is no data on categories (total uncertainty).
            - Returns 0 if there is only one category (complete certainty).
            - Calculates entropy using the given formula for all other cases.
    '''
    # Number of categories (n)
    n = len(categories)

    # Analyze the categories for each osmid
    n_osmid = n_land_use.get(osmid, 1)  # Avoid division by 0 by using a default value
    if n_osmid == 0:
        return 1  # Assign entropy of 1 if the category is uncertain
    if n_osmid == 1:
        return 0  # If there is only one category, entropy is 0
    
    # Get the area percentage (A_k/A_T) of each use for this osmid
    pk_values = porcentaje_usos_edges.loc[osmid]
    if isinstance(pk_values, pd.Series):
        pk_values = pk_values.values  # Convert to array if there is only one entry

    # Calculate entropy
    entropia = (-1 / np.log(n)) * np.sum([p * np.log(p) for p in pk_values if p > 0])

    return entropia


