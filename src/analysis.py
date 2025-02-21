import geopandas as gpd
import osmnx as ox
import igraph as ig
from rtree import index
import shapely as sh
from shapely.geometry import LineString, Point, Polygon, MultiLineString
import momepy
import numpy as np
from scipy.spatial import Voronoi
import pandas as pd
from shapely import ops



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


def create_network(nodes, edges, projected_crs="EPSG:6372",expand_coords=False):
    """
    Creates a network from nodes and edges without unique ids and to - from attributes by using coordinates.
    Assigs new unique 'key's for edges whenever there are duplicates in the 'u', 'v' and 'key' columns.

    Arguments:
        nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes for network in EPSG:4326.
        edges (geopandas.GeoDataFrame): GeoDataFrame with edges for network in EPSG:4326.
        projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".
        expand_coords (bool, optional): Boolean that, if true, multiplies coordinates by 10 to diminish the possibility of two nodes having the same osmid.

    Returns:
        geopandas.GeoDataFrame: nodes GeoDataFrame with unique ids based on coordinates named osmid in EPSG:4326
        geopandas.GeoDataFrame: edges GeoDataFrame with to - from attributes based on nodes ids named u and v respectively in EPSG:4326
    """

    # 1.1 --------------- LOAD AND PREPARE INPUT DATA
    #Copy edges and nodes to avoid editing original GeoDataFrames
    nodes = nodes.copy()
    edges = edges.copy()
    #Change coordinate system to meters for unique ids creation
    nodes = nodes.to_crs(projected_crs)
    edges = edges.to_crs(projected_crs)

    # 1.2 --------------- CREATE UNIQUE IDs BASED ON COORDINATES
    # Create unique id for nodes based on coordinates
    if expand_coords:
        nodes['osmid'] = (((nodes.geometry.x)*100).astype(int)).astype(str)+(((nodes.geometry.y)*100).astype(int)).astype(str)
    else:
        nodes['osmid'] = ((nodes.geometry.x).astype(int)).astype(str)+((nodes.geometry.y).astype(int)).astype(str)
    ##Set columns in edges for to[u] and from[v] columns
    edges['u'] = ''
    edges['v'] = ''
    edges.u.astype(str)
    edges.v.astype(str)
    # Create unique id for edges based on coordinates
    for index, row in edges.iterrows():
        if expand_coords:
            edges.at[index,'u'] = str(int((list(row.geometry.coords)[0][0])*100))+str(int((list(row.geometry.coords)[0][1])*100))
            edges.at[index,'v'] = str(int((list(row.geometry.coords)[-1][0])*100))+str(int((list(row.geometry.coords)[-1][1])*100))
        else:
            edges.at[index,'u'] = str(int(list(row.geometry.coords)[0][0]))+str(int(list(row.geometry.coords)[0][1]))
            edges.at[index,'v'] = str(int(list(row.geometry.coords)[-1][0]))+str(int(list(row.geometry.coords)[-1][1]))

    # 1.3 --------------- RE-REGISTER DUPLICATED EDGES BASED ON 'u'+'v'+'key'
    # Remove redundant nodes
    # nodes, edges = remove_redundant_nodes(nodes, edges)
    #Add key column for compatibility with OSMnx
    edges['key'] = 0
    
    # Code substituted by function resolve_duplicates_indexes()
    old_way = """
    # Find 'u', 'v' and 'key' duplicates in edges (Should never be the case)
    duplicated_edges = edges[edges.duplicated(subset=['u', 'v', 'key'], keep=False)]
    # Prepare registration_dict. Will hold unique 'u','v' and 'key' assigned.
    registration_dict = {}
    # For each duplicated edge found:
    for index,row in duplicated_edges.iterrows():
        # Obtain current 'u'+'v'
        # current_u = row['u']
        # current_v = row['v']
        u_v_id = str(row['u'])+str(row['v'])
        # If current 'u' and 'v' are already registered
        if u_v_id in registration_dict:
            # Read key that has been assigned
            registered_key = registration_dict[u_v_id]
            # Create new unregistered unique key
            new_key = registered_key+1
            # Register new unique key and update dictionary
            edges.loc[index,'key'] = new_key
            registration_dict[u_v_id] = new_key
            # print(f"Re-registered edge with u {current_u} and v {current_v} with key {new_key}.")
        # Else, it is the first time that this 'u' and 'v' is registered
        else:
            # Register new unique key and update dictionary
            edges.loc[index,'key'] = 0
            registration_dict[u_v_id] = 0
            # print(f"Re-registered edge with u {current_u} and v {current_v} with key 0.")"""

    # 1.4 --------------- FINAL OUTPUT FORMAT
    # Add x, y columns to nodes
    nodes['x'] = nodes.geometry.x
    nodes['y'] = nodes.geometry.y
    #Change [u,v] columns to integer
    edges['u'] = edges.u.astype(int)
    edges['v'] = edges.v.astype(int)
    #Calculate edges lentgh
    edges['length'] = edges.to_crs(projected_crs).length
    #Change osmid to integer
    nodes['osmid'] = nodes.osmid.astype(int)
    #Remove edge's unique ID ('u','v','key') duplicates
    edges = resolve_duplicates_indexes(edges, projected_crs)
    edges = edges.set_index(['u','v','key'])
    nodes = nodes.set_index('osmid')

    #Transform coordinates
    nodes = nodes.to_crs("EPSG:4326")
    edges = edges.to_crs("EPSG:4326")

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
    # # Simplify the tesellations
    # tessellation_gdf['geometry'] = tessellation_gdf['geometry'].simplify(tolerance=0.2, preserve_topology=True)
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
    # edges_consolidated['id'] = edges_consolidated.index
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
    # edges_consolidated['id'] = edges_consolidated.index
    edges_consolidated = edges_consolidated.set_crs(buildings.crs, allow_override=True)

    return nodes_consolidated, edges_consolidated, tessellation_gdf#, G, G_simplified


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


def network_from_tessellation(gdf, crs):
    """
    Generates a road network graph from a tessellation of the provided geometric data.

    Args:
        gdf (geopandas.GeoDataFrame): A GeoDataFrame containing the initial polygons or multipolygons from which the tessellation 
        will be derived. These geometries will be processed to generate road network features.
    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the nodes (points) of the generated road network.
    
    geopandas.GeoDataFrame: A GeoDataFrame containing the edges (lines) of the generated road network.
    """
    # gdf preprocessing
    gdf = gdf.to_crs(crs)
    gdf = gdf.copy()
    gdf = gdf.dissolve().explode().reset_index() # eliminate overlaping polygons and multipolygon
    # create tessellation
    limit = momepy.buffered_limit(gdf, buffer=50)
    tess_gdf = momepy.morphological_tessellation(gdf, clip=limit)
    # polygons to lines
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
    # extract first and last vertices from lines
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
    # format nodes and edges
    nodes, edges = network_entities(nodes_gdf, lines_single, crs=crs, expand_coords=(True,100))
    # delete unnecessary variables to free memory
    del nodes_gdf
    del lines_single
    # nodes, edges = create_network(nodes_gdf, lines_single, projected_crs=crs, expand_coords=True)
    #create graph
    G = ox.graph_from_gdfs(nodes, edges)
    # consolidate graph
    G2 = ox.consolidate_intersections(G, rebuild_graph=True, tolerance=10, dead_ends=True)
    # delete unnecessary variables to free memory
    del G
    # extract nodes and edges
    nodes, edges = ox.graph_to_gdfs(G2)

    # format nodes and edges
    nodes = nodes.reset_index()
    nodes = nodes.drop(columns=['osmid'])
    nodes = nodes.rename(columns={'osmid_original':'osmid'})
    nodes = nodes.set_index('osmid')
    edges = edges.reset_index()
    edges = edges.drop(columns=['u','v','index'])
    edges = edges.rename(columns={'u_original':'u',
    'v_original':'v'})
    edges = edges.set_index(['u','v','key'])

    return nodes, edges

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

def lines_connect(line1, line2):
    """ This function takes as input two lines (From a MultiLineString) and checks if they connect properly,
         if one needs to be reversed or if both need to be reversed.
	Args:
		line1 (geometry): Geometry of line 1 from the MultiLineString.
        line2 (geometry): Geometry of line 2 from the MultiLineString.
	Returns:
        connection (bool):True if both lines connect, False if they don't.
    	joining_line: The first line, as it is or reversed.
        new_line: The second line, as it is or reversed.
    """
    # Case 1: Last coord of first line connects with first coord of second line. No modification needed.
    if line1.coords[-1] == line2.coords[0]:
        return True, line1, line2
    
    # Case 2: First coord of first line connects with first coord of second line. Reverse first line.
    elif line1.coords[0] == line2.coords[0]:
        line1_reversed = LineString(line1.coords[::-1])
        return True, line1_reversed, line2

    # Case 3: Last coord of first line connects with Last coord of second line. Reverse second line.
    elif line1.coords[-1] == line2.coords[-1]:
        line2_reversed = LineString(line2.coords[::-1])
        return True, line1, line2_reversed

    # Case 4: First coord of first line connects with Last coord of second line. Reverse both lines.
    elif line1.coords[0] == line2.coords[-1]:
        line1_reversed = LineString(line1.coords[::-1])
        line2_reversed = LineString(line2.coords[::-1])
        return True, line1_reversed, line2_reversed

    # Case 5: No coords connect
    else:
        return False, line1, line2
    

def multilinestring_to_linestring(row):
    """ This function converts a MultiLineStrings to properly connected LineStrings.
	Args:
		row (geopandas.GeoDataFrame): row of a GeoDataFrame containing either a LineString or a MultiLineString in its geometry.
	Returns:
        geopandas.GeoDataFrame: row of a GeoDataFrame with no MultiLineStrings, LineStrings only.
    """
    line = row['geometry']
    
    # If the geometry is already a LineString, return it as is
    if isinstance(line, LineString):
        #print(f"Edge {row['edge_id']} is a LineString.")
        return row
    
    # If it's a MultiLineString, concatenate all LineStrings' coordinates, ensuring they connect
    elif isinstance(line, MultiLineString):
        #print(f"Edge {row['edge_id']} is a MultiLineString.")
        
        # Extract and combine all coordinates from each LineString in MultiLineString
        all_coords = list(line.geoms[0].coords)  # Start with the first LineString's coordinates

        # LineStrings to be concatenated to first LineString
        lines_i = []
        for i in range(1,len(line.geoms)):
            lines_i.append(i)

        # Added attempts limit for cases where an node is shared by two edges
        # that coincide in various points (very very rare, due to Volvo's Tessellations network)
        attempts = 0
        attempts_limit = 100
        
        # Iterate over the remaining LineStrings and ensure they connect
        while len(lines_i)>0: # While there are still lines to be connected

            if attempts < attempts_limit:
                # Iterate over each one of them and try to find a connection to line formed so far
                for i in lines_i: 

                    # Lines to connect
                    joining_line = LineString(all_coords) # The line formed so far
                    new_line = line.geoms[i] # The new line to be connected
                    # Check if lines connect, and reverse lines if needed for connection
                    connection, joining_line, new_line = lines_connect(joining_line, new_line)
                    # Perform connection
                    if connection:
                        # Register current coords (cannot use previous, might be reversed)
                        all_coords = list(joining_line.coords)
                        # Extend the coordinate list with the new line's coordinates
                        all_coords.extend(list(new_line.coords))
                        # Remove added i from lines_i list
                        lines_i.remove(i)
                    # Add attempt
                    attempts+=1
            else:
                # Stop
                print(f"Edge {row['edge_id']} exceeded the attempts limit for MultiLineString to LineString conversion.")
                global multilinestring_fail_lst
                multilinestring_fail_lst.append(row['edge_id'])
                return row
        
        # Update the row's geometry with the resulting LineString
        row['geometry'] = LineString(all_coords)
        return row

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

    # Find the number of edges that reach that osmid
    for osmid in list(nodes.osmid.unique()):
        # Total times = Times where that osmid is an edges 'u' + Times where that osmid is an edges 'v'
        streets_count = u_list.count(osmid) + v_list.count(osmid)
        # Data registration
        nodes.loc[nodes.osmid == osmid,'streets_count'] = streets_count

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


def resolve_duplicates_indexes(gdf, crs):
    """
    Resolves duplicates in a GeoDataFrame based on the multi-level index ('u', 'v', 'key') and a 'length' column.
    
    Parameters:
    gdf (geopandas.GeoDataFrame): The input GeoDataFrame with a multi-level index ('u', 'v', 'key') and a 'length' column.
        
    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame where duplicates based on the index are resolved according to the rules above.
    """
    
    # First, sort by index to ensure consistent grouping
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
