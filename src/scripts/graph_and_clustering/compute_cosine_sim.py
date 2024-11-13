from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import json
from pathlib import Path
from memory_profiler import profile
from scipy.sparse import csr_matrix, vstack
import gc
from tqdm import tqdm
import sys
from datetime import datetime
import time

class ProgressLogger:
    def __init__(self, log_file='cosine_similarity_progress.log'):
        self.log_file = log_file
        self.start_time = time.time()
        
    def log(self, message):
        """Log a message with timestamp to both file and stdout"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        log_message = f"[{timestamp}] [{elapsed_str}] {message}"
        
        print(log_message, flush=True)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
            f.flush()

class CosineSimilarityCalculator:
    def __init__(self, chunk_size=1000, output_dir='./outputs'):
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = ProgressLogger()
        self.checkpoint_file = self.output_dir / 'checkpoint.json'

    def save_checkpoint(self, i, j):
        """Save current progress to checkpoint file"""
        checkpoint = {
            'last_i': i,
            'last_j': j,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        self.logger.log(f"Checkpoint saved at chunk pair ({i}, {j})")

    def load_checkpoint(self):
        """Load the last checkpoint if it exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            self.logger.log(f"Found checkpoint from {checkpoint['timestamp']}")
            return checkpoint['last_i'], checkpoint['last_j']
        return None, None

    def estimate_memory_usage(self, n_samples, n_features):
        """Estimate memory usage for the computation"""
        chunk_memory = (self.chunk_size * n_features * 4) / (1024 ** 3)
        result_chunk = (self.chunk_size * self.chunk_size * 4) / (1024 ** 3)
        total_size = (n_samples * n_samples * 4) / (1024 ** 3)
        
        self.logger.log(f"Estimated memory usage per chunk: {chunk_memory:.2f} GB")
        self.logger.log(f"Estimated memory usage per result chunk: {result_chunk:.2f} GB")
        self.logger.log(f"Total similarity matrix size: {total_size:.2f} GB")

    def load_data(self, word_file, count_file):
        """Load and prepare input data"""
        self.logger.log("Loading selected words from JSON...")
        with open(word_file, 'r') as f:
            selected_words = list(json.load(f).keys())
        
        self.logger.log(f"Loading data from CSV... ({count_file})")
        df = pd.read_csv(count_file, usecols=['beer_id'] + selected_words)
        self.logger.log(f"Loaded {len(df)} rows and {len(selected_words)} selected words")
        
        return selected_words, df

    def normalize_matrix(self, X):
        """Normalize the matrix for cosine similarity calculation"""
        self.logger.log("Normalizing matrix...")
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        norms[norms == 0] = 1
        normalized = X / norms[:, np.newaxis]
        self.logger.log("Matrix normalization complete")
        return normalized

    def calculate_partial_similarity(self, chunk1, chunk2):
        """Calculate cosine similarity between two chunks"""
        density1 = (chunk1 != 0).mean()
        density2 = (chunk2 != 0).mean()
        
        if density1 < 0.1:
            chunk1 = csr_matrix(chunk1)
        if density2 < 0.1:
            chunk2 = csr_matrix(chunk2)
            
        return cosine_similarity(chunk1, chunk2)
    
    def save_metadata(self, beer_ids, output_path):
        """Save metadata including beer IDs and their mapping to matrix indices"""
        index_mapping = {
            'beer_ids': beer_ids.tolist(),
            'matrix_shape': (len(beer_ids), len(beer_ids)),
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'index_to_beer': dict(enumerate(beer_ids.tolist())),
            'beer_to_index': dict(zip(beer_ids.tolist(), range(len(beer_ids))))
        }
        
        with open(output_path / 'matrix_metadata.json', 'w') as f:
            json.dump(index_mapping, f, indent=2)
            
        self.logger.log(f"Saved metadata for {len(beer_ids)} beers")

    def process_chunks(self, data, beer_ids):
        """Process data in chunks to calculate cosine similarity"""
        n_samples = len(data)
        n_chunks = (n_samples + self.chunk_size - 1) // self.chunk_size
        total_chunks = (n_chunks * (n_chunks + 1)) // 2

        self.logger.log(f"Processing {n_samples} samples in {n_chunks} chunks")
        self.logger.log(f"Total chunk pairs to process: {total_chunks}")

        # Check for existing similarity matrix
        similarity_matrix_path = self.output_dir / 'similarity_matrix.npy'
        if not similarity_matrix_path.exists():
            # Create new memory-mapped file
            similarity_matrix = np.memmap(similarity_matrix_path, 
                                        dtype='float32', 
                                        mode='w+', 
                                        shape=(n_samples, n_samples))
            # Save initial metadata
            self.save_metadata(beer_ids, self.output_dir)
        else:
            # Open existing memory-mapped file
            similarity_matrix = np.memmap(similarity_matrix_path,
                                        dtype='float32',
                                        mode='r+',
                                        shape=(n_samples, n_samples))
            self.logger.log("Continuing with existing similarity matrix")

        # Load checkpoint if it exists
        checkpoint_i, checkpoint_j = self.load_checkpoint()
        
        # Normalize once
        self.logger.log("Starting matrix normalization...")
        normalized_data = self.normalize_matrix(data)
        self.logger.log("Matrix normalization complete")

        # Initialize progress tracking
        chunk_pairs_processed = 0
        total_remaining = total_chunks
        
        # If resuming from checkpoint, adjust progress
        if checkpoint_i is not None:
            processed_chunks = sum(n_chunks - j for j in range(checkpoint_i)) + (n_chunks - checkpoint_j)
            chunk_pairs_processed = processed_chunks
            total_remaining = total_chunks - processed_chunks
            self.logger.log(f"Resuming from checkpoint: chunk pair ({checkpoint_i}, {checkpoint_j})")

        progress_bar = tqdm(total=total_remaining, desc="Processing chunk pairs")

        try:
            for i in range(n_chunks):
                # Skip already processed chunks based on checkpoint
                if checkpoint_i is not None and i < checkpoint_i:
                    continue
                    
                start_i = i * self.chunk_size
                end_i = min((i + 1) * self.chunk_size, n_samples)
                chunk_i = normalized_data[start_i:end_i]

                self.logger.log(f"Processing chunk {i+1}/{n_chunks}")
                
                for j in range(i, n_chunks):
                    # Skip already processed chunks based on checkpoint
                    if checkpoint_i is not None and i == checkpoint_i and j < checkpoint_j:
                        continue
                    
                    start_j = j * self.chunk_size
                    end_j = min((j + 1) * self.chunk_size, n_samples)
                    chunk_j = normalized_data[start_j:end_j]

                    # Calculate similarity for this chunk pair
                    similarity_chunk = self.calculate_partial_similarity(chunk_i, chunk_j)

                    # Store in memmap file
                    similarity_matrix[start_i:end_i, start_j:end_j] = similarity_chunk
                    if i != j:  # Mirror the matrix
                        similarity_matrix[start_j:end_j, start_i:end_i] = similarity_chunk.T

                    # Update progress and save checkpoint
                    chunk_pairs_processed += 1
                    progress_bar.update(1)

                    # Save checkpoint and flush to disk periodically
                    if chunk_pairs_processed % 10 == 0:
                        self.save_checkpoint(i, j)
                        similarity_matrix.flush()

                # Reset checkpoint_i and checkpoint_j after first checkpoint row is complete
                if checkpoint_i is not None and i == checkpoint_i:
                    checkpoint_i = None
                    checkpoint_j = None

                # Clear memory
                gc.collect()
                self.logger.log(f"Memory cleaned after chunk {i+1}")

        except Exception as e:
            self.logger.log(f"Error during processing: {str(e)}")
            # Save checkpoint before raising exception
            self.save_checkpoint(i, j)
            raise
        finally:
            progress_bar.close()
            similarity_matrix.flush()
            del similarity_matrix
            
            # If completed successfully, remove checkpoint file
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                self.logger.log("Computation completed successfully, checkpoint file removed")
        
    @profile
    def run(self, word_file, count_file):
        """Main execution method"""
        self.logger.log("Starting cosine similarity calculation")
        
        # Load data
        selected_words, df = self.load_data(word_file, count_file)
        
        # Estimate memory usage
        self.estimate_memory_usage(len(df), len(selected_words))
        
        # Store beer IDs
        beer_ids = df['beer_id'].values
        
        # Convert to numpy array for faster processing
        self.logger.log("Converting DataFrame to numpy array...")
        data = df[selected_words].values
        self.logger.log("Conversion complete")
        
        # Process chunks
        self.logger.log("Starting chunk processing...")
        self.process_chunks(data, beer_ids)
        
        self.logger.log("Computation complete. Results stored in numpy memmap format.")

# Helper functions remain the same
def load_similarity_matrix(output_dir):
    output_dir = Path(output_dir)
    
    with open(output_dir / 'matrix_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    matrix_shape = tuple(metadata['matrix_shape'])
    similarity_matrix = np.memmap(output_dir / 'similarity_matrix.npy',
                                dtype='float32',
                                mode='r',
                                shape=matrix_shape)
    
    return similarity_matrix, metadata

def get_similar_beers(beer_id, n=5, output_dir='./outputs'):
    matrix, metadata = load_similarity_matrix(output_dir)
    
    try:
        idx = metadata['beer_to_index'][str(beer_id)]
    except KeyError:
        raise ValueError(f"Beer ID {beer_id} not found in the dataset")
    
    similarities = matrix[idx]
    similar_indices = np.argsort(similarities)[-n-1:-1][::-1]
    
    similar_beers = [
        (metadata['index_to_beer'][str(idx)], float(similarities[idx]))
        for idx in similar_indices
    ]
    
    return similar_beers

def main():
    calculator = CosineSimilarityCalculator(chunk_size=1000)
    
    try:
        calculator.run('selected_words.json', 'beer_word_counts2.csv')
        
        print("\nExample usage:", flush=True)
        print("To find similar beers, use:", flush=True)
        print("similar_beers = get_similar_beers(beer_id, n=5)", flush=True)
        
    except Exception as e:
        calculator.logger.log(f"Error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()