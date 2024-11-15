#!/usr/bin/env python

import os
import pandas as pd
import igraph as ig
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pickle
import json
import kaleido
from mpi4py import MPI
from pathlib import Path
from datetime import datetime
import psutil
import gc
import logging
import traceback
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import os
from typing import Dict, List, Tuple, Any
import logging


logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

def log_message(msg, rank=None, flush=True):
    """Log a message with timestamp and rank information."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rank_str = f"[Rank {rank}]" if rank is not None else "[Main]"
    print(f"{timestamp} {rank_str} {msg}", flush=flush)

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    mem_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    return f"{mem_gb:.2f}GB"

def load_similarity_matrix(output_dir, rank, ratio=1.0):
    """Load memory-mapped similarity matrix and metadata."""
    try:
        output_dir = Path(output_dir)
        
        log_message(f"Loading matrix metadata from {output_dir}", rank)
        with open(output_dir / 'matrix_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        matrix_shape = tuple(metadata['matrix_shape'])
        num_rows_cols = int(matrix_shape[0] * ratio)
        
        log_message(f"Memory usage before matrix mapping: {get_memory_usage()}", rank)
        similarity_matrix = np.memmap(output_dir / 'similarity_matrix.npy',
                                    dtype='float32',
                                    mode='r',
                                    shape=matrix_shape)
        
        similarity_matrix = similarity_matrix[:num_rows_cols, :num_rows_cols]
        log_message(f"Memory usage after matrix mapping: {get_memory_usage()}", rank)
        
        metadata['matrix_shape'] = (num_rows_cols, num_rows_cols)
        
        return similarity_matrix, metadata
    except Exception as e:
        log_message(f"Error loading similarity matrix: {str(e)}", rank)
        raise

def get_gradient_color(color1: str, color2: str, ratio: float) -> Tuple[float, float, float]:
    """Calculate gradient color between two hex colors."""
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    c1 = hex_to_rgb(color1)
    c2 = hex_to_rgb(color2)
    return tuple(c1[i] * (1 - ratio) + c2[i] * ratio for i in range(3))

def get_node_ratings(G, result_df):
    """Get node ratings and names from the graph and result dataframe.
    
    Args:
        G: igraph Graph object with 'label' vertex attributes
        result_df: DataFrame with columns ['beer_id', 'total_ratings', 'beer_name']
    
    Returns:
        tuple: (node_ratings, node_names) lists containing ratings and names for each vertex
    """
    node_ratings = []
    node_names = []
    
    # Convert beer_id to int type if it isn't already
    result_df['beer_id'] = result_df['beer_id'].astype(int)
    
    for vertex in G.vs:
        try:
            beer_id = int(vertex["label"])
            # Find matching beer in dataframe
            matching_beers = result_df[result_df['beer_id'] == beer_id]
            
            if matching_beers.empty:
                # Handle case where beer_id isn't found
                logging.warning(f"Beer ID {beer_id} not found in result_df")
                node_ratings.append(0)  # or some default value
                node_names.append(f"Unknown Beer {beer_id}")
            else:
                # Take first matching beer
                beer_data = matching_beers.iloc[0]
                node_ratings.append(beer_data['total_ratings'])
                node_names.append(beer_data['beer_name'])
                
        except ValueError as e:
            # Handle case where vertex label isn't a valid integer
            logging.error(f"Invalid vertex label: {vertex['label']}")
            node_ratings.append(0)
            node_names.append(f"Invalid Beer {vertex['label']}")
            
        except Exception as e:
            # Handle any other unexpected errors
            logging.error(f"Error processing vertex {vertex['label']}: {str(e)}")
            node_ratings.append(0)
            node_names.append(f"Error: {vertex['label']}")
    
    return node_ratings, node_names

# Helper function to check data integrity before processing
def validate_beer_data(G, result_df):
    """Validate graph and DataFrame data before processing."""
    logging.info(f"Graph has {len(G.vs)} vertices")
    logging.info(f"DataFrame has {len(result_df)} rows")
    
    # Check required columns exist
    required_columns = ['beer_id', 'total_ratings', 'beer_name']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in DataFrame: {missing_columns}")
    
    # Check for vertex labels in DataFrame
    graph_beer_ids = set(int(v['label']) for v in G.vs)
    df_beer_ids = set(result_df['beer_id'].astype(int))
    
    missing_beers = graph_beer_ids - df_beer_ids
    if missing_beers:
        logging.warning(f"Found {len(missing_beers)} beers in graph but not in DataFrame: {missing_beers}")
    
    extra_beers = df_beer_ids - graph_beer_ids
    if extra_beers:
        logging.warning(f"Found {len(extra_beers)} beers in DataFrame but not in graph: {extra_beers}")
    
    return True

def validate_data(similarity_matrix, metadata, result_df_selected):
    """Validate input data before processing."""
    logging.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    logging.info(f"Metadata length: {len(metadata)}")
    logging.info(f"Result DataFrame shape: {result_df_selected.shape}")
    
    if result_df_selected.empty:
        raise ValueError("result_df_selected is empty")
    
    return True

def calculate_node_sizes(ratings):
    """Calculate node sizes based on ratings."""
    min_size = 5
    max_size = 50
    if len(ratings) == 0:
        return []
    log_ratings = np.log1p(ratings)
    normalized_sizes = (log_ratings - np.min(log_ratings)) / (np.max(log_ratings) - np.min(log_ratings))
    return min_size + normalized_sizes * (max_size - min_size)

def normalize_edge_weights(G):
    """Normalize edge weights for visualization."""
    weights = G.es['weight']
    if not weights:
        return []
    min_width = 0.1
    max_width = 2.0
    normalized_weights = np.array(weights)
    if np.max(normalized_weights) == np.min(normalized_weights):
        return [min_width] * len(weights)
    normalized_weights = (normalized_weights - np.min(normalized_weights)) / (np.max(normalized_weights) - np.min(normalized_weights))
    return min_width + normalized_weights * (max_width - min_width)

def stream_significant_edges_parallel(similarity_matrix, metadata, threshold=0.7, batch_size=100000, comm=None):
    """
    Stream significant edges in parallel using a strided row distribution with MPI error handling.
    Uses pickle serialization for MPI communication.
    
    Args:
        similarity_matrix: numpy.ndarray, square similarity matrix
        metadata: dict, metadata for the nodes including beer ID mappings
        threshold: float, minimum similarity threshold
        batch_size: int, number of edges to process before garbage collection
        comm: MPI.Comm, MPI communicator (optional)
    
    Returns:
        list of tuples: (source_id, target_id, source_beer_id, target_beer_id, similarity_score)
    """
    # Setup MPI environment
    try:
        rank = comm.Get_rank() if comm else 0
        size = comm.Get_size() if comm else 1
    except MPI.Exception as e:
        logging.error(f"MPI initialization failed: {str(e)}")
        raise
        
    # Configure logging
    logging.basicConfig(
        format='%(asctime)s - Rank %(message)s',
        level=logging.INFO
    )
    
    n = similarity_matrix.shape[0]
    edges = []
    
    # Get the index to beer mapping from metadata
    index_to_beer = metadata['index_to_beer']
    # log some index_to_beer values
    logging.info(f"Index to beer value keys : {list(index_to_beer.keys())[:5]}")
    logging.info(f"Index to beer value values : {list(index_to_beer.values())[:5]}")
    
    try:
        # Process rows in a strided manner
        logging.info(f"Rank {rank} processing {n} rows with stride {size}")
        for i in tqdm(range(rank, n, size), 
                     desc=f"Rank {rank} processing", 
                     disable=rank!=0):
            row_edges = []
            
            # Process each row in chunks to manage memory
            for j in range(i + 1, n, batch_size):
                chunk_end = min(j + batch_size, n)
                # Use vectorized operations for better performance
                mask = similarity_matrix[i, j:chunk_end] >= threshold
                indices = np.nonzero(mask)[0] + j
                
                for idx in indices:
                    row_edges.append((
                        str(i),
                        str(idx),
                        index_to_beer[str(i)],  # Add source beer ID
                        index_to_beer[str(idx)],  # Add target beer ID
                        float(similarity_matrix[i, idx])
                    ))
            
            edges.extend(row_edges)
            
            if len(edges) >= batch_size:
                gc.collect()
            
            if len(edges) % (batch_size * 10) == 0:
                logging.info(f"{rank}: Processed {len(edges)} edges so far")
    
        # Gather edges using pickle serialization
        logging.info(f"{rank}: Finished processing {len(edges)} total edges")
        if comm:
            try:
                # First, synchronize all processes
                comm.Barrier()
                
                # Serialize the edges using pickle
                serialized_edges = pickle.dumps(edges)
                
                # Gather the sizes first
                size_of_data = len(serialized_edges)
                all_sizes = comm.gather(size_of_data, root=0)
                
                if rank == 0:
                    logging.info(f"0: Received data sizes from all ranks")
                    
                    # Receive and deserialize data from each rank
                    all_edges = []
                    for i in range(size):
                        if i == 0:
                            # Root process already has its own data
                            rank_edges = pickle.loads(serialized_edges)
                        else:
                            # Receive serialized data from other ranks
                            rank_data = comm.recv(source=i, tag=0)
                            rank_edges = pickle.loads(rank_data)
                        all_edges.extend(rank_edges)
                        logging.info(f"0: Processed data from rank {i}")
                    
                    edges = all_edges
                    logging.info(f"0: Successfully gathered {len(edges)} total edges")
                else:
                    # Send serialized data to root
                    comm.send(serialized_edges, dest=0, tag=0)
                    edges = []
                
                # Final synchronization
                comm.Barrier()
                
            except MPI.Exception as e:
                logging.error(f"{rank}: MPI operation failed: {str(e)}")
                raise
            except Exception as e:
                logging.error(f"{rank}: Error during gather: {str(e)}")
                raise
                
    except Exception as e:
        logging.error(f"{rank}: Processing failed: {str(e)}")
        # traceback error
        logging.error(traceback.format_exc())
        if comm:
            comm.Abort(1)
        raise
        
    finally:
        # Clean up
        gc.collect()
        
    return edges


def process_edge_batch(batch_edges, G, pos, partition, community_colors, edge_widths, batch_start):
    """Process a batch of edges for visualization"""
    edge_traces = []
    
    # log some values
    logging.info(f"Batch edges : {batch_edges[:5]}")
    logging.info(f"Batch edges length : {len(batch_edges)}")
    logging.info(f"source 0: {batch_edges[0].source}")
    logging.info(f"target 0: {batch_edges[0].target}")
    logging.info(f"source 0 label: {G.vs['label'][int(batch_edges[0].source)]}")
    logging.info(f"target 0 label: {G.vs['label'][int(batch_edges[0].target)]}")
    # pos keys
    logging.info(f"pos keys : {list(pos.keys())[:5]}")
    

    for idx, edge in enumerate(batch_edges):
        try:
            source, target = str(edge.source), str(edge.target)
            
            source_beer_id = str(int(G.vs['label'][int(source)]))
            target_beer_id = str(int(G.vs['label'][int(target)]))

            x0, y0 = pos[source_beer_id]
            x1, y1 = pos[target_beer_id]

            color1 = community_colors[str(partition[source_beer_id])]
            color2 = community_colors[str(partition[target_beer_id])]
            edge_color = get_gradient_color(color1, color2, 0.5)
            
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=edge_widths[batch_start + idx], color=edge_color),
                    hoverinfo='none',
                    mode='lines',
                    opacity=0.2,
                    showlegend=False
                )
            )
        except (KeyError, IndexError) as e:
            log_message(f"Error processing edge {edge}: {str(e)}")
            # traceback
            log_message(traceback.format_exc())
            continue
    
    return edge_traces

def process_node_batch(batch_indices, G, pos, partition, community_colors, node_sizes, node_names, node_ratings):
    """Process a batch of nodes for visualization"""
    community_traces = {comm: {'x': [], 'y': [], 'sizes': [], 'texts': []} 
                       for comm in set(partition.values())}
    
    for idx in batch_indices:
        try:
            vertex = G.vs[idx]
            node = str(int(vertex["label"]))
            x, y = pos[node]
            community = partition[node]
            
            community_traces[community]['x'].append(x)
            community_traces[community]['y'].append(y)
            community_traces[community]['sizes'].append(node_sizes[idx])
            
            if node_sizes[idx] > 10:
                hover_text = f'Beer: {node_names[idx]}<br>ID: {node}<br>Ratings: {node_ratings[idx]:,}'
            else:
                hover_text = ''
            
            community_traces[community]['texts'].append(hover_text)
        except (KeyError, IndexError) as e:
            log_message(f"Error processing node {idx}: {str(e)}")
            continue
    
    return community_traces

def serialize_graph_data(G, layout, partition):
    """Serialize graph data with validation."""
    try:
        logging.info(f"Graph vertices: {len(G.vs)}, edges: {len(G.es)}")
        logging.info(f"Layout dict length: {len(layout)}")
        logging.info(f"Partition dict length: {len(partition)}")
        
        # Validate vertex labels exist
        if 'label' not in G.vs.attributes():
            raise ValueError("Graph vertices missing 'label' attribute")
            
        # Validate edge weights exist
        if 'weight' not in G.es.attributes():
            raise ValueError("Graph edges missing 'weight' attribute")
        
        essential_data = {
            'edges': [(int(e.source), int(e.target)) for e in G.es],
            'weights': G.es['weight'],
            'labels': [str(v['label']) for v in G.vs],
            'positions': {str(int(v['label'])): layout[v.index] for v in G.vs}
        }
        
        # Safely create positions dictionary
        # for v in G.vs:
        #     try:
        #         label = str(int(v['label']))
        #         if v.index in layout:
        #             essential_data['positions'][label] = layout[v.index]
        #         else:
        #             logging.warning(f"Missing position for vertex {v.index}")
        #     except (KeyError, ValueError) as e:
        #         logging.warning(f"Error processing vertex {v.index}: {str(e)}")
        
        return essential_data
    except Exception as e:
        logging.error(f"Error in serialize_graph_data: {str(e)}")
        raise

def reconstruct_graph(data):
    """Reconstruct graph from essential data."""
    G = ig.Graph()
    G.add_vertices(len(data['labels']))
    G.vs['label'] = data['labels']
    G.add_edges(data['edges'])
    G.es['weight'] = data['weights']
    return G

def get_main_component(G):
    subgraphs = G.decompose()

    logging.info(f"Found {len(subgraphs)} subgraphs")

    # log the length of each subgraph
    for i, sg in enumerate(subgraphs):
        logging.info(f"Subgraph {i} has {len(sg.vs)} vertices and {len(sg.es)} edges")
    
    # Get the largest connected component
    largest_subgraph = subgraphs[0]
    for sg in subgraphs:
        if len(sg.vs) > len(largest_subgraph.vs):
            largest_subgraph = sg

    logging.info(f"Largest subgraph has {len(largest_subgraph.vs)} vertices and {len(largest_subgraph.es)} edges")

    # verify connectivity
    is_connected = len(largest_subgraph.connected_components()) == 1

    logging.info(f"Is fully connected: {is_connected}")

    return largest_subgraph


def visualize_beer_network_parallel(similarity_matrix, metadata, result_df_selected, partition, output_dir='outputs', ratio=1.0, load_temp_data=False, calculate_layout=True):
    """
    Parallel version of beer network visualization using MPI.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if not load_temp_data:
        if rank == 0:
            log_message(f"Starting parallel visualization with {size} processes")
            os.makedirs(output_dir, exist_ok=True)
        
        # Build graph from similarity matrix
        edges = stream_significant_edges_parallel(
            similarity_matrix,
            metadata,
            threshold=0.7,
            batch_size=100000,
            comm=comm
        )
            
        if rank == 0:
            log_message("Building graph incrementally...")
            
            # Process edges in chunks to manage memory
            CHUNK_SIZE = 1000000  # Process 1M edges at a time
            
            # First pass: collect unique vertices and their corresponding beer IDs
            vertex_set = set()
            vertex_to_beer = {}  # Map vertex ID to beer ID
            for i in range(0, len(edges), CHUNK_SIZE):
                chunk = edges[i:i + CHUNK_SIZE]
                for e in chunk:
                    # e now contains (source_id, target_id, source_beer_id, target_beer_id, similarity)
                    vertex_set.add(e[0])
                    vertex_set.add(e[1])
                    # Map vertex IDs to beer IDs
                    vertex_to_beer[e[0]] = e[2]  # source
                    vertex_to_beer[e[1]] = e[3]  # target
                
                if i % (CHUNK_SIZE * 10) == 0:
                    log_message(f"Processed {i}/{len(edges)} edges for vertex collection")
                    gc.collect()
            
            vertices = sorted(list(vertex_set))
            vertex_dict = {v: i for i, v in enumerate(vertices)}
            
            # Create list of beer IDs in the same order as vertices
            beer_ids = [vertex_to_beer[v] for v in vertices]
            
            del vertex_set  # Free memory
            del vertex_to_beer  # Free memory
            gc.collect()
            
            log_message(f"Created vertex mapping for {len(vertices)} vertices")
            
            # Initialize graph
            G = ig.Graph()
            G.add_vertices(len(vertices))
            G.vs["label"] = beer_ids  # Use beer IDs as labels
            
            # Process edges in chunks
            for i in range(0, len(edges), CHUNK_SIZE):
                chunk = edges[i:i + CHUNK_SIZE]
                
                # Process each chunk
                chunk_edges = []
                chunk_weights = []
                
                for e in chunk:
                    try:
                        source_idx = vertex_dict[e[0]]
                        target_idx = vertex_dict[e[1]]
                        chunk_edges.append((source_idx, target_idx))
                        chunk_weights.append(e[4])  # similarity is now at index 4
                    except KeyError as ke:
                        log_message(f"Warning: Vertex not found in mapping: {ke}")
                        continue
                
                # Add edges and weights from this chunk
                if chunk_edges:
                    G.add_edges(chunk_edges)
                    G.es[-(len(chunk_edges)):]["weight"] = chunk_weights
                
                if i % (CHUNK_SIZE * 10) == 0:
                    log_message(f"Added {i}/{len(edges)} edges to graph")
                    gc.collect()
            
            del edges  # Free original edges list
            del vertex_dict  # Free the mapping dictionary
            gc.collect()
                    
            # Calculate layout
            log_message("Calculating layout...")
            # base_size = np.sqrt(len(G.vs)) * 5
            # box_size_x = base_size * 16
            # box_size_y = base_size * 9

            # Pre-calculate some graph metrics to inform parameters
            avg_degree = len(G.es) / len(G.vs)  # ~333 edges per node
            graph_density = 2 * len(G.es) / (len(G.vs) * (len(G.vs) - 1))  # ~0.11 density

            # Calculate optimal box size based on graph size and density
            # Larger box for dense graphs to give nodes more space
            base_box_size = np.sqrt(len(G.vs)) * 100  # Base size scaled by sqrt of node count
            density_factor = 1 + np.log10(1 + graph_density)  # Increase space for denser graphs
            box_size_x = base_box_size * density_factor
            box_size_y = box_size_x * 0.75  # Keep 4:3 aspect ratio

            if calculate_layout:
                G = get_main_component(G)

                layout = G.layout_fruchterman_reingold(
                    weights=G.es['weight'],
                    niter=3000,
                    start_temp=len(G.vs),
                    grid="nogrid",
                )

                # layout = G.layout_drl(
                #     weights=G.es['weight'],
                #     options="default"

                # )

                # Scale the layout to fill the box
                # Convert layout to numpy array for easier manipulation
                coords = np.array(layout)
                
                if len(coords) > 0:  # Check if graph is not empty
                    # Get current bounds
                    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
                    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
                    
                    # Calculate scaling factors
                    scale_x = (2 * box_size_x) / (max_x - min_x) if max_x != min_x else 1
                    scale_y = (2 * box_size_y) / (max_y - min_y) if max_y != min_y else 1
                    
                    # Apply scaling and centering
                    coords[:, 0] = (coords[:, 0] - min_x) * scale_x - box_size_x
                    coords[:, 1] = (coords[:, 1] - min_y) * scale_y - box_size_y
                    
                    # Convert back to list of tuples
                    layout = [tuple(coord) for coord in coords]
                else:
                    log_message("Empty graph after layout calculation")
                
                # Save layout to file
                layout_file = os.path.join(output_dir, f'layout_{ratio}.pkl')
                with open(layout_file, 'wb') as f:
                    pickle.dump(layout, f)
                log_message(f"Layout saved to {layout_file}")
            else :
                # Load layout from file
                layout_file = os.path.join(output_dir, f'layout_{ratio}.pkl')
                with open(layout_file, 'rb') as f:
                    layout = pickle.load(f)
                log_message(f"Layout loaded from {layout_file}")
            
            # Convert layout to dictionary
            # log some layout values
            logging.info(f"Layout value 0 : {layout[0]}")
            logging.info(f"Layout value 1 : {layout[1]}")
            # log some vertex values
            logging.info(f"Vertex value 0 : {G.vs[0]}")
            logging.info(f"Vertex value 0 index : {G.vs[0].index}")
            logging.info(f"Vertex value 0 label : {G.vs[0]['label']}")

            pos = {str(int(v['label'])): layout[v.index] for v in G.vs}
        else:
            G = None
            pos = None
        
        if rank == 0:
            try:
                # Validate input data
                validate_data(similarity_matrix, metadata, result_df_selected)
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Log important information before processing
                logging.info("Starting graph visualization process")
                logging.info(f"Number of communities: {len(set(partition.values()))}")
                
                # Serialize essential graph data
                logging.info("Preparing graph data for broadcasting...")
                essential_data = serialize_graph_data(G, layout, partition)
                
                # Setup colors and sizes with validation
                n_communities = len(set(partition.values()))
                palette = sns.color_palette("husl", n_communities).as_hex()
                community_colors = {str(comm): color for comm, color 
                                in zip(sorted(set(partition.values())), palette)}
                
                # Calculate node properties with additional logging
                logging.info("Calculating node properties...")
                node_ratings, node_names = get_node_ratings(G, result_df_selected)
                node_sizes = calculate_node_sizes(node_ratings)
                edge_widths = normalize_edge_weights(G)
                
                logging.info(f"Node ratings: {node_ratings[:5]}")
                logging.info(f"Node names: {node_names[:5]}")
                logging.info(f"Node sizes: {node_sizes[:5]}")
                logging.info(f"Edge widths: {edge_widths[:5]}")
                logging.info(f"partition keys : {list(partition.keys())[:5]}")
                logging.info(f"partition values : {list(partition.values())[:5]}")

                # Package all data for broadcasting
                # broadcast_data = {
                #     'graph_data': essential_data,
                #     'community_colors': community_colors,
                #     'node_ratings': {str(k): v for k, v in node_ratings.items()},
                #     'node_names': {str(k): v for k, v in node_names.items()},
                #     'node_sizes': {str(k): v for k, v in node_sizes.items()},
                #     'edge_widths': edge_widths,
                #     'partition': {str(k): v for k, v in partition.items()}
                # }
                broadcast_data = {
                    'graph_data': essential_data,
                    'community_colors': community_colors,
                    'node_ratings': node_ratings,
                    'node_names': node_names,
                    'node_sizes': node_sizes,
                    'edge_widths': edge_widths,
                    'partition': {str(k): v for k, v in partition.items()}
                }
                
                # Save data with error handling
                temp_file = os.path.join(output_dir, f'temp_broadcast_data_{ratio}.pkl')
                try:
                    with open(temp_file, 'wb') as f:
                        pickle.dump(broadcast_data, f)
                    logging.info(f"Successfully saved broadcast data to {temp_file}")
                except Exception as e:
                    logging.error(f"Failed to save broadcast data: {str(e)}")
                    raise
                
                broadcast_message = {'temp_file': temp_file}
                
            except Exception as e:
                logging.error(f"Error preparing data: {str(e)}")
                comm.Abort(1)
                raise
        else:
            broadcast_message = None
    else:
        if rank == 0:
            # load f'temp_broadcast_data_{ratio}.pkl'
            temp_file = os.path.join(output_dir, f'temp_broadcast_data_{ratio}.pkl')
            logging.info(f"Loading broadcast data from {temp_file}")
            broadcast_message = {'temp_file': temp_file}
            logging.info(f"Loaded.")
        else:
            broadcast_message = None
    
    # Broadcast the file path
    broadcast_message = comm.bcast(broadcast_message, root=0)
    
    # All processes load data from file
    try:
        with open(broadcast_message['temp_file'], 'rb') as f:
            broadcast_data = pickle.load(f)
        
        # Reconstruct graph and data structures
        G = reconstruct_graph(broadcast_data['graph_data'])
        pos = broadcast_data['graph_data']['positions']
        community_colors = broadcast_data['community_colors']
        node_ratings = broadcast_data['node_ratings']
        node_names = broadcast_data['node_names']
        node_sizes = broadcast_data['node_sizes']
        edge_widths = broadcast_data['edge_widths']
        partition = broadcast_data['partition']
        
        # Clear broadcast_data from memory since we've extracted what we need
        del broadcast_data
        gc.collect()
        
    except Exception as e:
        log_message(f"Error loading data on rank {rank}: {str(e)}")
        comm.Abort(1)
    
    # Clean up temporary file
    # if rank == 0:
    #     try:
    #         os.remove(broadcast_message['temp_file'])
    #     except:
    #         pass
    
    # Process edges and nodes in parallel
    fig = create_network_visualization(
        G, pos, partition, community_colors, 
        node_sizes, node_names, node_ratings,
        edge_widths, comm, rank, size, output_dir, ratio=ratio
    )
    
    return fig
    
def create_network_visualization(G, pos, partition, community_colors, 
                               node_sizes, node_names, node_ratings,
                               edge_widths, comm, rank, size, output_dir, ratio=1.0):
    """Create and save network visualization using matplotlib with MPI parallelization."""
    edge_data = process_edges_parallel(G, pos, partition, community_colors, 
                                     edge_widths, comm, rank, size)
    
    if rank == 0:
        return create_and_save_figure(
            G, edge_data, pos, partition, community_colors, 
            node_sizes, edge_widths, output_dir, ratio
        )
    return None

def process_edges_parallel(G, pos, partition, community_colors, edge_widths, comm, rank, size):
    """Process edges in parallel using MPI."""
    total_edges = len(G.es)
    
    # Calculate local workload for this process
    edges_per_rank = total_edges // size + (1 if rank < total_edges % size else 0)
    start_idx = rank * (total_edges // size) + min(rank, total_edges % size)
    end_idx = start_idx + edges_per_rank
    
    # Initialize local storage
    local_edge_coords = []
    local_edge_colors = []
    local_edge_widths = []
    
    # Process edges in batches
    batch_size = 10000
    num_local_batches = (edges_per_rank + batch_size - 1) // batch_size
    
    if rank == 0:
        logging.info(f"Processing {total_edges:,} edges across {size} processes")
    
    # Create progress bar only for rank 0
    pbar = None
    if rank == 0:
        pbar = tqdm(total=edges_per_rank, desc=f"Process {rank} processing edges")
    
    for batch_start in range(start_idx, end_idx, batch_size):
        batch_end = min(batch_start + batch_size, end_idx)
        batch_edges = G.es[batch_start:batch_end]
        
        for idx, edge in enumerate(batch_edges):
            try:
                source, target = str(edge.source), str(edge.target)
                source_beer_id = str(int(G.vs['label'][int(source)]))
                target_beer_id = str(int(G.vs['label'][int(target)]))

                x0, y0 = pos[source_beer_id]
                x1, y1 = pos[target_beer_id]
                
                color1 = community_colors[str(partition[source_beer_id])]
                color2 = community_colors[str(partition[target_beer_id])]
                edge_color = get_gradient_color(color1, color2, 0.5)
                
                local_edge_coords.append([(x0, y0), (x1, y1)])
                local_edge_colors.append(edge_color)
                local_edge_widths.append(edge_widths[batch_start - start_idx + idx])
                
                if pbar is not None:
                    pbar.update(1)
                    
            except (KeyError, IndexError) as e:
                if rank == 0:
                    logging.error(f"Error processing edge {edge}: {str(e)}")
                continue
    
    if pbar is not None:
        pbar.close()
    
    # Gather results from all processes
    all_edge_coords = comm.gather(local_edge_coords, root=0)
    all_edge_colors = comm.gather(local_edge_colors, root=0)
    all_edge_widths = comm.gather(local_edge_widths, root=0)
    
    if rank == 0:
        # Combine results from all processes
        edge_coords = [coord for process_coords in all_edge_coords for coord in process_coords]
        edge_colors = [color for process_colors in all_edge_colors for color in process_colors]
        edge_widths = [width for process_widths in all_edge_widths for width in process_widths]
        
        return {
            'coords': edge_coords,
            'colors': edge_colors,
            'widths': edge_widths
        }
    return None

def create_and_save_figure(G, edge_data, pos, partition, community_colors, 
                          node_sizes, edge_widths, output_dir, ratio=1.0):
    """Create and save the final figure using matplotlib."""
    try:
        logging.info("Starting visualization creation...")
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 9), facecolor='#1E1E1E')
        ax.set_facecolor('#1E1E1E')

        # Draw all edges at once using LineCollection
        logging.info("Drawing edges...")
        edge_collection = LineCollection(edge_data['coords'], 
                                      colors=edge_data['colors'],
                                      linewidths=edge_data['widths'],
                                      alpha=0.2,
                                      zorder=1)
        ax.add_collection(edge_collection)

        # Process nodes by community
        communities = set(partition.values())
        logging.info(f"Processing {len(communities)} communities...")
        
        # log some values
        logging.info(f"Node sizes : {node_sizes[:5]}")
        logging.info(f"Pos keys : {list(pos.keys())[:5]}")
        logging.info(f"Pos values : {list(pos.values())[:5]}")
        logging.info(f"Partition keys : {list(partition.keys())[:5]}")
        logging.info(f"Partition values : {list(partition.values())[:5]}")
        logging.info(f"G.vs['label'] : {G.vs['label'][:5]}")
        
        # check if '28248' in partition keys
        logging.info(f"'28248' in partition keys : {'28248' in partition.keys()}")
        # check if '262145' in pos keys
        logging.info(f"'262145' in pos keys : {'262145' in pos.keys()}")

        # get max and min values of partition keys (converted to int)
        logging.info(f"Max partition key : {max([int(k) for k in partition.keys()])}")
        logging.info(f"Min partition key : {min([int(k) for k in partition.keys()])}")
        # get max and min values of pos keys (converted to int)
        logging.info(f"Max pos key : {max([int(k) for k in pos.keys()])}")
        logging.info(f"Min pos key : {min([int(k) for k in pos.keys()])}")
        # get max and min values of G.vs['label'] (converted to int)
        logging.info(f"Max G.vs['label'] : {max([int(G.vs['label'][k]) for k in range(len(G.vs['label']))])}")
        logging.info(f"Min G.vs['label'] : {min([int(G.vs['label'][k]) for k in range(len(G.vs['label']))])}")
        # get length of G.vs['label']
        logging.info(f"Length of G.vs['label'] : {len(G.vs['label'])}")



        with tqdm(total=len(communities), desc="Processing communities") as pbar:
            for community in communities:
                community_nodes = [node_id for node_id, comm in partition.items() 
                                 if str(comm) == str(community) and node_id in pos]
                
                logging.info(f"community nodes : {community_nodes[:5]}")

                if not community_nodes:
                    continue

                logging.info(f"Processing community {community} with {len(community_nodes):,} nodes")
                
                x_coords = [pos[node_id][0] for node_id in community_nodes]
                y_coords = [pos[node_id][1] for node_id in community_nodes]
                node_sizes_comm = [node_sizes[G.vs.find(label=node_id).index] 
                                 for node_id in community_nodes]
                
                ax.scatter(x_coords, y_coords,
                          s=node_sizes_comm,
                          c=[community_colors[str(community)]],
                          linewidth=0.5,
                          edgecolor='#444444',
                          label=f'Community {community}',
                          zorder=2)
                pbar.update(1)

        logging.info("Finalizing visualization...")
        
        # Set layout
        ax.set_title(f'Beer Network Communities ({len(G.vs):,} beers)',
                    color='white', pad=20, size=24)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add legend
        legend = ax.legend(bbox_to_anchor=(1.05, 0.5),
                          loc='center left',
                          title='Communities',
                          facecolor='#1E1E1E',
                          edgecolor='#444444')
        legend.get_title().set_color('white')
        for text in legend.get_texts():
            text.set_color('white')

        # Adjust layout and save
        plt.tight_layout()
        
        logging.info("Saving visualization...")
        output_path = os.path.join(output_dir, f'beer_network_{ratio}.png')
        fig.savefig(output_path,
                   dpi=300,
                   bbox_inches='tight',
                   facecolor='#1E1E1E',
                   edgecolor='none')
        plt.close(fig)
        
        logging.info(f"Visualization saved successfully to {output_path}")
        logging.info(f"Memory usage stats: {get_memory_usage()}")

    except Exception as e:
        logging.error(f"Error creating figure: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank==0:
            logging.info("Starting MPI process...")
        
        ratio_beers = 0.1
        load_temp_data = False
        calculate_layout = True

        if not load_temp_data:
            if rank == 0:
                # Load data
                try:
                    log_message("Loading input data...", rank)
                    result_df = pd.read_csv('beer_word_counts2.csv', 
                                        usecols=['beer_id', 'total_ratings'])
                    with open('beer_name_dict.pkl', 'rb') as f:
                        beer_name_dict = pickle.load(f)
                    result_df["beer_name"] = result_df["beer_id"].map(beer_name_dict)
                    
                    # Load partition
                    with open('outputs/partition.json', 'r') as f:
                        partition = json.load(f)
                except Exception as e:
                    log_message(f"Error loading input data: {str(e)}")
                    raise
            else:
                result_df = None
                partition = None
            
            # Load similarity matrix
            similarity_matrix, metadata = load_similarity_matrix('outputs', rank, ratio=ratio_beers)
            
            # Broadcast necessary data
            result_df = comm.bcast(result_df, root=0)
            partition = comm.bcast(partition, root=0)
            
        else:
            log_message(f"Rank {rank} loading temporary data", rank)
            result_df = None
            partition = None
            similarity_matrix = None
            metadata = None
        # Visualize the network
        fig = visualize_beer_network_parallel(
            similarity_matrix,
            metadata,
            result_df,
            partition,
            output_dir='outputs',
            ratio=ratio_beers,
            load_temp_data=load_temp_data,
            calculate_layout=calculate_layout
        )
        
    except Exception as e:
        log_message(f"Critical error: {str(e)}", rank)
        log_message(traceback.format_exc(), rank)
        raise
    
    finally:
        MPI.Finalize()
