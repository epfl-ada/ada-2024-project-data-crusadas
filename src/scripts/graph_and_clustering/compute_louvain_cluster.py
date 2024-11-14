#!/usr/bin/env python

# file : compute_louvain_cluster.py

import pandas as pd
import igraph as ig
import numpy as np
from pathlib import Path
import json
from itertools import combinations
import gc
from tqdm import tqdm
import argparse
from mpi4py import MPI
import sys
from datetime import datetime
import psutil

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

def parse_args():
    parser = argparse.ArgumentParser(description='Cluster similarity matrix using MPI')
    parser.add_argument('--output-dir', required=True, help='Directory containing similarity matrix')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    parser.add_argument('--batch-size', type=int, default=100000, help='Batch size for processing')
    parser.add_argument('--resolution', type=float, default=1.0, help='Louvain resolution parameter')
    return parser.parse_args()

def load_similarity_matrix(output_dir, rank, ratio=1.0):
    """Load memory-mapped similarity matrix and metadata, keeping only a ratio of columns and rows."""
    output_dir = Path(output_dir)
    
    log_message(f"Loading matrix metadata from {output_dir}", rank)
    with open(output_dir / 'matrix_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    matrix_shape = tuple(metadata['matrix_shape'])
    log_message(f"Matrix shape: {matrix_shape}", rank)
    
    # Calculate the number of rows and columns to keep based on the ratio
    num_rows_cols = int(matrix_shape[0] * ratio)
    log_message(f"Keeping {num_rows_cols} rows and columns based on ratio {ratio}", rank)
    
    log_message(f"Memory usage before matrix mapping: {get_memory_usage()}", rank)
    similarity_matrix = np.memmap(output_dir / 'similarity_matrix.npy',
                                dtype='float32',
                                mode='r',
                                shape=matrix_shape)
    
    # Keep only the first num_rows_cols rows and columns
    similarity_matrix = similarity_matrix[:num_rows_cols, :num_rows_cols]
    log_message(f"Memory usage after matrix mapping: {get_memory_usage()}", rank)
    
    # Update metadata to reflect the reduced matrix size
    metadata['matrix_shape'] = (num_rows_cols, num_rows_cols)
    
    return similarity_matrix, metadata

def stream_significant_edges_parallel(similarity_matrix, metadata, threshold=0.5, 
                                   batch_size=10000, comm=None):
    """Parallel version of edge streaming using MPI with round-robin row distribution."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    total_rows = similarity_matrix.shape[0]
    
    # Select rows for this rank in a round-robin fashion
    assigned_rows = list(range(rank, total_rows, size))
    log_message(f"Rank {rank} processing rows {assigned_rows[:10]}... to {assigned_rows[-10:]}", rank)
    
    local_edges = []
    beer_ids = metadata.get('beer_ids', list(range(total_rows)))
    
    # Track progress through assigned rows instead of chunks
    total_assigned_rows = len(assigned_rows)
    last_progress = 0
    edges_found = 0
    
    # Process all assigned rows
    for row_num, row_idx in enumerate(assigned_rows):
        row_beer_id = beer_ids[row_idx]
        values = similarity_matrix[row_idx, :]
        
        local_row_edges = 0
        for col_idx in range(row_idx + 1, total_rows):
            if values[col_idx] >= threshold:
                local_edges.append((row_beer_id, beer_ids[col_idx], float(values[col_idx])))
                local_row_edges += 1
        
        edges_found += local_row_edges
        
        # Log progress every 5%
        progress = ((row_num + 1) * 100) // total_assigned_rows
        if progress >= last_progress + 5:
            log_message(f"Progress: {progress}% complete ({row_num + 1}/{total_assigned_rows} rows, "
                       f"{edges_found} edges found), Memory: {get_memory_usage()}", rank)
            last_progress = progress
            gc.collect()
    
    log_message(f"Completed edge detection. Total edges found: {len(local_edges)}", rank)
    
    # Gather edge counts first to log total
    edge_counts = comm.gather(len(local_edges), root=0)
    if rank == 0:
        total_edges = sum(edge_counts)
        log_message(f"Total edges across all processes: {total_edges}")
    
    # Gather edges from all processes
    log_message(f"Gathering edges from all processes", rank)
    all_edges = comm.gather(local_edges, root=0)
    log_message(f"Edge gathering complete", rank)
    
    if rank == 0:
        return [edge for process_edges in all_edges for edge in process_edges]
    return None

def cluster_with_louvain(edges, metadata, resolution=1.0, random_state=None):
    """Apply Louvain clustering using igraph with progress tracking."""
    log_message("Preparing graph for Louvain clustering")
    log_message(f"Initial memory usage: {get_memory_usage()}")
    
    # Create vertex list with beer IDs
    vertices = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))
    vertex_map = {vid: idx for idx, vid in enumerate(vertices)}
    
    # Create igraph Graph object
    log_message("Building igraph Graph...")
    G = ig.Graph()
    G.add_vertices(len(vertices))
    
    # Add vertex labels (beer IDs)
    G.vs["label"] = vertices
    
    # Convert edges to use vertex indices and add weights
    log_message("Converting edges...")
    edge_list = [(vertex_map[e[0]], vertex_map[e[1]]) for e in edges]
    weights = [e[2] for e in edges]
    
    # Add edges to graph
    log_message("Adding edges to graph...")
    G.add_edges(edge_list)
    G.es["weight"] = weights
    
    log_message(f"Graph built with {G.vcount()} vertices and {G.ecount()} edges")
    log_message(f"Memory usage after graph construction: {get_memory_usage()}")
    
    # Run Louvain clustering
    log_message("Applying Louvain clustering...")
    partition = G.community_multilevel(weights="weight")
    log_message("Clustering complete")
    
    # Convert partition to dictionary format
    partition_dict = {vertices[idx]: membership for idx, membership in enumerate(partition.membership)}
    
    return G, partition_dict

def main_mpi():
    """MPI-enabled main function with progress tracking."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        log_message(f"Starting clustering pipeline with {size} processes")
        args = parse_args()
    else:
        args = None
    
    args = comm.bcast(args, root=0)
    
    # Load matrix and metadata
    if rank == 0:
        log_message("Loading similarity matrix on root node")
        similarity_matrix, metadata = load_similarity_matrix(args.output_dir, rank)
        matrix_size = similarity_matrix.shape[0]
        log_message(f"Matrix size: {matrix_size}x{matrix_size}")
    else:
        similarity_matrix = metadata = None
    
    # Broadcast metadata
    metadata = comm.bcast(metadata, root=0)
    
    # Each process loads its own view of the memory-mapped matrix
    if rank != 0:
        similarity_matrix, _ = load_similarity_matrix(args.output_dir, rank)
    
    # Synchronize processes before starting edge detection
    comm.Barrier()
    if rank == 0:
        log_message("All processes ready, starting edge detection")
    
    # Parallel edge generation
    edges = stream_significant_edges_parallel(
        similarity_matrix,
        metadata,
        threshold=args.threshold,
        batch_size=args.batch_size,
        comm=comm
    )
    
    if rank == 0:
        log_message(f"Edge detection complete. Building graph from {len(edges)} edges")
        log_message(f"Memory usage before graph construction: {get_memory_usage()}")
        
        # Build graph and perform clustering
        G, partition = cluster_with_louvain(edges, metadata, resolution=args.resolution)
        
        # Save results
        output_file = Path(args.output_dir) / 'graph.graphml'
        log_message(f"Saving graph to {output_file}")
        G.write_graphml(str(output_file))
        
        partition_file = Path(args.output_dir) / 'partition.json'
        log_message(f"Saving partition to {partition_file}")
        with open(partition_file, 'w') as f:
            json.dump(partition, f)
        
        log_message(f"Final memory usage: {get_memory_usage()}")
    
    comm.Barrier()
    if rank == 0:
        log_message("All processes completed successfully")
    
    MPI.Finalize()

if __name__ == "__main__":
    main_mpi()