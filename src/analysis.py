import geopandas as gpd
import osmnx as ox
import igraph as ig
from rtree import index
import shapely as sh
from shapely.geometry import LineString, Point, Polygon, MultiLineString
import momepy
import numpy as np
from scipy.spatial import Voronoi
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import pandas as pd
from shapely import ops

# Used in notebook 01_PL_02_Boeing_network.ipynb
def get_boeing_network(zone_blocks:gpd.GeoDataFrame, nodes:gpd.GeoDataFrame, edges:gpd.GeoDataFrame, buff:int):
    
    '''
    Function to extract the nodes and edges of Boeing network for a territory considering a buffer.
    Blocks must be in cartesian coordinates.

    Args:
        zone_blocks (gpd.GeoDataFrame): Blocks of the analysis zone represented by Polygons.
        nodes (gpd.GeoDataFrame): Nodes of Boeing network represented by Points.
        edges (gpd.GeoDataFrame): Edges of Boeing network represented by LineStrings.
        buff (int): Radius of the buffer.
    Reuturns:
        zone_boeing_nodes (gpd.GeoDataFrame): Boeing nodes of the analysis zone plus the buffer.
        zone_boeing_edges (gpd.GeoDataFrame): Boeing edges of the analysis zone plus the buffer.
    '''

    # Usar unary_union para fusionar todos los polígonos en una geometría única
    poligono_exterior = zone_blocks.union_all().convex_hull.buffer(buff)

    # Crear un nuevo GeoDataFrame con el polígono exterior
    zona = gpd.GeoDataFrame(geometry=gpd.GeoSeries([poligono_exterior]), crs=zone_blocks.crs)
    zona = zona.to_crs('epsg:4326')

    # Set crs
    nodes = nodes.to_crs('epsg:4326')
    edges = edges.to_crs('epsg:4326')

    # Filtrar los edges de boeing de la zona
    zone_boeing_edges = gpd.sjoin(edges, zona, predicate='intersects')
    zone_boeing_edges = zone_boeing_edges.drop(['index_right'], axis=1)

    # Obtener los nodos que corresponden a los edges filtrados
    filtered_node_ids = set(zone_boeing_edges['from']).union(set(zone_boeing_edges['to']))

    # Filtrar los nodos de boeing de la zona
    zone_boeing_nodes = nodes[nodes['ID'].isin(filtered_node_ids)]

    return zone_boeing_nodes,zone_boeing_edges

# Used in notebook 01_PL_04_Combine_networks.ipynb
# Used in function network_from_tessellation()
def network_entities(nodes:gpd.GeoDataFrame, edges:gpd.GeoDataFrame, crs='epsg:32618', expand_coords=(False, 10)):
    
    """
    Create a network based on nodes and edges without unique ids and to - from attributes.

    Args:
        nodes (gpd.GeoDataFrame): GeoDataFrame with nodes for network in EPSG:4326
        edges (gpd.GeoDataFrame): GeoDataFrame with edges for network in EPSG:4326
        crs (str): Desired output cordinate system

    Returns:
        nodes (gpd.GeoDataFrame): nodes GeoDataFrame with unique ids based on coordinates named osmid in desired crs
        edges (gpd.GeoDataFrame): edges GeoDataFrame with to - from attributes based on nodes ids named u and v respectively in desired crs
    """

    # Copy edges and nodes to avoid editing original GeoDataFrames
    nodes = nodes.copy()
    edges = edges.copy()
    # Change coordinate system to meters for unique ids
    nodes = nodes.to_crs(crs)
    edges = edges.to_crs(crs)
    # Unique str id for nodes based on coordinates
    print('Creating unique ids (osmid) for nodes based on coordinates...')
    if expand_coords[0]:
        nodes['osmid'] = (((nodes.geometry.x)*expand_coords[1]).astype(int)).astype(str)+(((nodes.geometry.y)*expand_coords[1]).astype(int)).astype(str)
    else:
        nodes['osmid'] = ((nodes.geometry.x).astype(int)).astype(str)+((nodes.geometry.y).astype(int)).astype(str)
    ##Set columns in edges for to[u] and from[v] columns
    print('Creating unique ids (u,v) for edges based on coordinates...')
    edges['u'] = ''
    edges['v'] = ''
    edges.u.astype(str)
    edges.v.astype(str)
    # Create unique id for edges based on coordinates
    for index, row in edges.iterrows():
        if expand_coords[0]:
            edges.at[index,'u'] = str(int((list(row.geometry.coords)[0][0])*expand_coords[1]))+str(int((list(row.geometry.coords)[0][1])*expand_coords[1]))
            edges.at[index,'v'] = str(int((list(row.geometry.coords)[-1][0])*expand_coords[1]))+str(int((list(row.geometry.coords)[-1][1])*expand_coords[1]))
        else:
            edges.at[index,'u'] = str(int(list(row.geometry.coords)[0][0]))+str(int(list(row.geometry.coords)[0][1]))
            edges.at[index,'v'] = str(int(list(row.geometry.coords)[-1][0]))+str(int(list(row.geometry.coords)[-1][1]))
    # Remove redundant nodes
    print('Removing redundant nodes...')
    nodes, edges = remove_redundant_nodes(nodes, edges)
    # Add key column for compatibility with osmnx
    edges['key'] = 0
    # Calculate edges lentgh
    edges['length'] = edges.to_crs(crs).length
    # Organice nodes and edges
    nodes['x'] = nodes['geometry'].x
    nodes['y'] = nodes['geometry'].y

    # remove duplicates
    print('Resolving indexes u, v, key...')
    edges = resolve_duplicates_indexes(edges, crs)
    edges = edges.drop_duplicates(subset=['u','v','key'])
    nodes = nodes.drop_duplicates(subset=['osmid'])
    edges = edges.set_index(['u','v','key'])
    nodes = nodes.set_index('osmid')

    return nodes, edges

# Used in notebook 02_PV_01_Slope_DEM.ipynb
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

# Used in notebook 02_PV_02_Population.ipynb
# Used in notebook 02_PV_02b_Population (voronoi_points_within_aoi()).ipynb
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

# Used in notebook 01_PL_03_Tessellation_network.ipynb
# Used in script 01-create-tessellations-network.py
def network_from_tessellation(gdf, crs, consolidate=(True,10)):
    """
    Generates a road network graph from a tessellation of the provided geometric data.

    Args:
        gdf (geopandas.GeoDataFrame): A GeoDataFrame containing the initial polygons or multipolygons from which the tessellation 
                                      will be derived. These geometries will be processed to generate road network features.
        crs (str): The coordinate reference system to be used for the generated road network.
        consolidate (tuple, optional): A tuple containing a boolean value and a float value. If the boolean value is True, the
                                       generated road network will be consolidated. The float value represents the tolerance for consolidation. 
                                       Defaults to (True,10).
    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the nodes (points) of the generated road network.
    
    geopandas.GeoDataFrame: A GeoDataFrame containing the edges (lines) of the generated road network.
    """
    
    # gdf preprocessing
    gdf = gdf.to_crs(crs)
    gdf = gdf.copy()
    gdf = gdf.dissolve().explode().reset_index() # eliminate overlaping polygons and multipolygon
    
    # Create tessellation
    print('Creating tessellation...')
    limit = momepy.buffered_limit(gdf, buffer=50)
    tess_gdf = momepy.morphological_tessellation(gdf, clip=limit)
    # polygons to lines
    print('Converting polygons to lines...')
    lines_gdf = gpd.GeoDataFrame(geometry=tess_gdf.geometry.boundary)
    # delete unnecessary variables to free memory
    del tess_gdf
    del gdf
    del limit
    # separate lines by intersection
    lines_single = gpd.GeoDataFrame(geometry=lines_gdf.dissolve().geometry.map(lambda x: ops.linemerge(x)).explode())
    lines_single = lines_single.set_crs(crs)
    lines_single = lines_single.reset_index(drop=True)
    lines_single = lines_single.reset_index()

    # Extract first and last vertices from lines
    print('Extracting points from lines...')
    point_geo = [Point(lines_single.iloc[i].geometry.coords[0]) for i in range(len(lines_single))]
    point_geo.extend([Point(lines_single.iloc[i].geometry.coords[-1]) for i in range(len(lines_single))])
    # remove duplicates
    point_geo_filter = set(point_geo)
    point_geo_filter = list(point_geo_filter)
    # create gdf from point geometries
    nodes_gdf = pd.DataFrame(point_geo_filter)
    nodes_gdf = nodes_gdf.rename(columns={0:'geometry'})
    nodes_gdf = nodes_gdf.set_geometry('geometry')
    nodes_gdf = nodes_gdf.set_crs(crs)

    # Format nodes and edges
    print('Creating nodes and edges...')
    # nodes, edges = create_network(nodes_gdf, lines_single, projected_crs=crs, expand_coords=True)
    nodes, edges = network_entities(nodes_gdf, lines_single, crs=crs, expand_coords=(True,100))
    # delete unnecessary variables to free memory
    del nodes_gdf
    del lines_single
    
    # Consolidate if required
    if consolidate[0]:
        # Create graph
        print('Creating graph...')
        G = ox.graph_from_gdfs(nodes, edges)
        print(f'Consolidating graph using tolerance of {consolidate[1]} meters...')
        # consolidate graph
        G2 = ox.consolidate_intersections(G, rebuild_graph=True, tolerance=consolidate[1], dead_ends=True)
        del G #Save space
        # Extract nodes and edges from consolidated graph
        nodes, edges = ox.graph_to_gdfs(G2)
        del G2 #Save space
        # Format nodes and edges
        print('Formating nodes and edges...')
        nodes = nodes.reset_index()
        nodes = nodes.drop(columns=['osmid'])
        nodes = nodes.rename(columns={'osmid_original':'osmid'})
        nodes = nodes.set_index('osmid')
        edges = edges.reset_index()
        edges = edges.drop(columns=['u','v'])
        edges = edges.rename(columns={'u_original':'u',
        'v_original':'v'})
        edges = edges.set_index(['u','v','key'])

    if 'index' in nodes.columns:
        nodes = nodes.drop(columns=['index'])
    if 'index' in edges.columns:
        edges = edges.drop(columns=['index'])

    return nodes, edges

# Used in notebook 01_PL_04_Combine_networks.ipynb
def create_unique_edge_id(edges, order='uvkey'):
    """
    Create a unique edge_id based on the 'u', 'v' and 'key' columns of the edges GeoDataFrame.

    Args:
        edges (geopandas.GeoDataFrame): GeoDataFrame containing the edges of the network.
        order (str, optional): Order for the unique id. Defaults to 'uvkey'.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with the unique edge_id column.
    """
    # Turn ID data to string
    edges['u'] = edges['u'].astype('str')
    edges['v'] = edges['v'].astype('str')
    edges['key'] = edges['key'].astype('str')
    # Concatenate ID data to create unique edge_id
    if order == 'uvkey':
        edges['edge_id'] = edges['u']+edges['v']+edges['key']
    elif order == 'vukey':
        edges['edge_id'] = edges['v']+edges['u']+edges['key']
    # Turn ID data back to int
    edges['u'] = edges['u'].astype('int')
    edges['v'] = edges['v'].astype('int')
    edges['key'] = edges['key'].astype('int')

    return edges

# Used in functions network_entities() and create_network()
def remove_redundant_nodes(nodes:gpd.GeoDataFrame, edges:gpd.GeoDataFrame):
    """
    Remove nodes that only are connected to two edges.

    Args:
        nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes for network in EPSG:4326
        edges (geopandas.GeoDataFrame): GeoDataFrame with edges for network in EPSG:4326

    Returns:
        geopandas.GeoDataFrame: nodes GeoDataFrame without redundant nodes
        geopandas.GeoDataFrame: edges GeoDataFrame with merged edges and without redundant nodes
    """

    # Check if 'u' and 'v' are in the columns of edges and 'osmid' in nodes or are part of index
    if 'u' not in edges.columns:
         edges = edges.reset_index()

    if 'osmid' not in nodes.columns:
        nodes = nodes.reset_index()
    
    # Extract the 'u' and 'v' columns from edges
    u_list = list(edges.u)
    v_list = list(edges.v)

    # Calculate street count for each node
    old_way = """
    # Find the number of edges that reach that osmid
    for osmid in list(nodes.osmid.unique()):
        # Total times = Times where that osmid is an edges 'u' + Times where that osmid is an edges 'v'
        streets_count = u_list.count(osmid) + v_list.count(osmid)
        # Data registration
        nodes.loc[nodes.osmid == osmid,'streets_count'] = streets_count
    """
    # OPTIMIZED WAY:
    # Count osmid occurrences and store as a pd.Series (fillna as 0s if a given osmid is misssing from 'u' or 'v')
    u_counts = pd.Series(u_list).value_counts().fillna(0)
    v_counts = pd.Series(v_list).value_counts().fillna(0)
    # Add counts from both series 
    streets_count_series = (u_counts.add(v_counts, fill_value=0)).astype(int)
    # Assign values to joined_nodes_cleaning using map
    nodes['streets_count'] = nodes['osmid'].map(streets_count_series).fillna(0).astype(int)

    # Select nodes with two streets
    two_edges_osmids = list(nodes.loc[nodes.streets_count==2].osmid.unique())

    # multilinestring_fail_lst = []

    for osmid in two_edges_osmids:

        # try:

        #print("--"*10)
        #print(f"OSMID OF INTEREST: {osmid}.")
        
        # Find edges that use that osmid
        found_edges = edges.loc[(edges.u == osmid)|(edges.v == osmid)].copy()
        # found_in_v = edges.loc[edges.v == osmid].copy()
        # found_edges = pd.concat([found_in_u,found_in_v])
        #print(found_edges)
        
        # Find the other osmids those edges connect with
        u_v_list = list(found_edges.u.unique()) + list(found_edges.v.unique())

        # Remove itself
        u_v_list = [i for i in u_v_list if i != osmid]
        # If both edges connect to the same osmid (It is a loop road split in two)
        # Double that osmid
        if len(u_v_list) == 1:
            u_v_list.append(u_v_list[0])
        elif len(u_v_list) == 0:
            continue
        
        # Dissolve lines (Creates MultiLineString, will convert to LineString)
        flattened_edge = found_edges.dissolve()
        # Flatten MultiLineString to LineString
        flattened_edge["geometry"] = flattened_edge["geometry"].apply(ops.linemerge)
    
        # Add data to new edge
        flattened_edge['u'] = u_v_list[0]
        flattened_edge['v'] = u_v_list[1]
        flattened_edge['key'] = 0

        # Delete useless node and previous edges, concat new flattened edge.
        nodes = nodes.loc[nodes.osmid != osmid].copy()
        edges = edges.loc[(edges.u != osmid)&(edges.v != osmid)].copy()
        edges = pd.concat([edges,flattened_edge])

        # flattened_edge['ntw_origin'] = 'ntw_cleaning'
        '''flattened_edge = create_unique_edge_id(flattened_edge)
    
        # Convert MultiLineStrings to LineStrings
        flattened_edge = flattened_edge.apply(multilinestring_to_linestring,axis=1)
        # If conversion fails, flattened edge_id gets added to global list multilinestring_fail_lst.
        flattened_edge_id = flattened_edge.edge_id.unique()[0]
        if flattened_edge_id not in multilinestring_fail_lst:
            # Delete useless node and previous edges, concat new flattened edge.
            nodes = nodes.loc[nodes.osmid != osmid].copy()
            edges = edges.loc[(edges.u != edges)&(edges.v != osmid)].copy()
            edges = pd.concat([edges,flattened_edge])
        else:
            print(f"Not dissolving edges reaching node {osmid}.")

    except:
        print(osmid)
        print(u_v_list)'''

    # Final format
    # nodes.reset_index(inplace=True,drop=True)
    # edges.reset_index(inplace=True,drop=True)

    return nodes, edges

# Used in functions network_entities() and create_network()
def resolve_duplicates_indexes(gdf, crs):
    """
    Resolves duplicates in a GeoDataFrame based on the multi-level index ('u', 'v', 'key') and a 'length' column.
    
    Parameters:
    gdf (geopandas.GeoDataFrame): The input GeoDataFrame with a multi-level index ('u', 'v', 'key') and a 'length' column.
        
    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame where duplicates based on the index are resolved according to the rules above.
    """
    
    # First, sort by index to ensure consistent grouping
    gdf.reset_index(inplace=True)
    if 'index' in gdf.columns:
        gdf.drop(columns=['index'],inplace=True)
    gdf = gdf.sort_index()
    
    # Group by the multi-level index ('u', 'v', 'key')
    grouped = gdf.groupby(['u', 'v', 'key'])
    
    # Lists to track rows to drop and new rows with modified keys
    rows_to_drop = []
    new_rows = []
    
    for (u, v, key), group in grouped:
        if len(group) > 1:
            # Check if 'length' values are the same for all rows in this group
            if group['length'].nunique() == 1:
                # If the 'length' is the same for all rows, drop the duplicates, keeping the first
                rows_to_drop.append(group.index[1:])  # Keep the first, drop the rest
            else:
                # If 'length' is different, increment the 'key' of each of the following rows by 1
                change_key=0
                for i in range(1, len(group)):
                    change_key+=1
                    new_row = group.iloc[i].copy() # Copy the row
                    new_row['key'] = change_key # Increment the key
                    new_rows.append(new_row) # Append the new row
                    rows_to_drop.append([group.index[i]]) # Drop the original row
    
    # Drop the identified duplicate rows
    gdf = gdf.drop(pd.Index([index for sublist in rows_to_drop for index in sublist]))
    
    # Add the new rows with the incremented 'key'
    # gdf = pd.DataFrame(gdf) # set as DataFrame for concat
    gdf = pd.concat([gdf, pd.DataFrame(new_rows)], ignore_index=False)

    # Set geometry
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=crs)
    
    # Return the modified DataFrame sorted by the index
    return gdf

# Used in notebook 02_PV_02b_Project_network_Voronois.ipynb
def voronoi_points_within_aoi(area_of_interest, points, points_id_col, admissible_error=0.01, projected_crs="EPSG:6372"):
	""" Creates voronoi polygons within a given area of interest (aoi) from n given points.
	Args:
		area_of_interest (geopandas.GeoDataFrame): GeoDataFrame with area of interest (Determines output extents).
		points (geopandas.GeoDataFrame): GeoDataFrame with points of interest.
		points_id_col (str): Name of points ID column (Will be assigned to each resulting voronoi polygon)
		admissible_error (int, optional): Percentage of error (difference) between the input area (area_of_interest) and output area (dissolved voronoi polygons).
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".
	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with voronoi polygons (each containing the point ID it originated from) extending all up to the area of interest extent.
	"""

	# Set area of interest and points of interest for voronoi analysis to crs:6372 (Proyected)
	aoi = area_of_interest.to_crs(projected_crs)
	pois = points.to_crs(projected_crs)

    # Distance is a number used to create a buffer around the polygon and coordinates along a bounding box of that buffer.
    # Starts at 100 (works for smaller polygons) but will increase itself automatically until the diference between the area of 
    # the voronoi polygons created and the area of the aoi is less than the admissible_error.
	distance = 100

    # Goal area (Area of aoi)
	# Objective is that diff between sum of all voronois polygons and goal area is within admissible error.
	goal_area_gdf = aoi.copy()
	goal_area_gdf['area'] = goal_area_gdf.geometry.area
	goal_area = goal_area_gdf['area'].sum()
	
	# Kick start while loop by creating area_diff 
	area_diff = admissible_error + 1 
	while area_diff > admissible_error:
		# Create a rectangular bound for the area of interest with a {distance} buffer.
		polygon = aoi['geometry'].unique()[0]
		bound = polygon.buffer(distance).envelope.boundary
		
		# Create points along the rectangular boundary every {distance} meters.
		boundarypoints = [bound.interpolate(distance=d) for d in range(0, np.ceil(bound.length).astype(int), distance)]
		boundarycoords = np.array([[p.x, p.y] for p in boundarypoints])
		
		# Load the points inside the polygon
		coords = np.array(pois.get_coordinates())
		
		# Create an array of all points on the boundary and inside the polygon
		all_coords = np.concatenate((boundarycoords, coords))
		
		# Calculate voronoi to all coords and create voronois gdf (No boundary)
		vor = Voronoi(points=all_coords)
		lines = [sh.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
		polys = sh.ops.polygonize(lines)
		unbounded_voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=projected_crs)

		# Add nodes ID data to voronoi polygons
		unbounded_voronois = gpd.sjoin(unbounded_voronois,pois[[points_id_col,'geometry']])
		unbounded_voronois = unbounded_voronois.drop(columns=['index_right'])
        
		# Clip voronoi with boundary
		bounded_voronois = gpd.overlay(df1=unbounded_voronois, df2=aoi, how='intersection')

		# Change back crs
		voronois_gdf = bounded_voronois.to_crs('EPSG:4326')

		# Area check for while loop
		voronois_area_gdf = voronois_gdf.to_crs(projected_crs)
		voronois_area_gdf['area'] = voronois_area_gdf.geometry.area
		voronois_area = voronois_area_gdf['area'].sum()
		area_diff = ((goal_area - voronois_area)/(goal_area))*100
		if area_diff > admissible_error:
			print(f'Error = {round(area_diff,2)}%. Repeating process.')
			distance = distance * 10
		else:
			print(f'Error = {round(area_diff,2)}%. Admissible.')

	# Out of the while loop:
	return voronois_gdf

# Used in function calculate_density()
def epanechnikov_kernel(dist, bandwidth):
    #Originaly located in utils/analysis.py
    """
    This function implements the Epanechnikov kernel for kernel density estimation, defining the weight of a point based on its distance from the center.
    
    Args:
        dist (float): The distance from the center of the kernel to the point.
        bandwidth (float): The bandwidth of the kernel, which defines the radius of influence.
    
    Returns:
        float: The weight of the point based on the Epanechnikov kernel function.
    """
    return 0.75 * (1 - (dist / bandwidth) ** 2) if dist < bandwidth else 0

# Used in function calculate_density()
def quartic_kernel(dist, bandwidth):
    #Originaly located in utils/analysis.py
    """
    This function implements the Quartic (biweight) kernel for kernel density estimation, 
    defining the weight of a point based on its distance from the center.

    Args:
        dist (float): The distance from the center of the kernel to the point.
        bandwidth (float): The bandwidth of the kernel, which defines the radius of influence.
    
    Returns:
        float: The weight of the point based on the Quartic kernel function.
    """
    return (15 / 16) * ((1 - (dist / bandwidth) ** 2) ** 2) if dist < bandwidth else 0

# Used in notebook 02_PLV_03_Itersections.ipynb
def calculate_density(points, bandwidth, pixel_size, kernel_shape):
    #Originaly located in utils/analysis.py
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

# Used in notebook 02_PLV_03_Itersections.ipynb, but it is commented
def plot_density(points, density):
    #Originaly located in utils/analysis.py
    """
    This function plots the density of points on a 2D grid using a heatmap representation.

    Args:
        points (gpd.GeoDataFrame): A GeoDataFrame containing the points for which density is calculated.
            Must include a 'geometry' column with Point geometries.
        density (np.ndarray): A 2D NumPy array representing the density grid, with rows and columns corresponding 
            to the y and x coordinates of the grid.
    
    Returns:
        None: Displays a heatmap of the density.
    """

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

# Used in function 02_PV_05_Land_use.ipynb
def calcular_entropia(osmid, n_land_use, porcentaje_usos_edges, categories):
    #Originaly located in utils/analysis.py
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
        categories (list): List of land use categories to consider in the entropy calculation.

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