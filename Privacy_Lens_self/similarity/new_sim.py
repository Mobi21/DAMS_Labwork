import os
import json
import re
import time
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Hypothetical Ollama Python interface
import ollama  

###############################################################################
# 1. LOGGING CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("embedding_experiment.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

###############################################################################
# 2. HELPER FUNCTIONS FOR READING & WRITING DATA
###############################################################################
def load_dataset(filename="data.json", text_col="policy_text"):
    """
    Loads a single JSON file of policies into a pandas DataFrame.
    The JSON should look like:
    [
      {"policy_text": "Some privacy policy text 1", "id": "unique1", ...},
      {"policy_text": "Some privacy policy text 2", "id": "unique2", ...},
      ...
    ]
    """
    logger.info(f"Loading dataset from {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Ensure the required text column is present
    if filename != "similarity_matrix.json":
        if text_col not in df.columns:
            raise ValueError(f"Expected column '{text_col}' not found in JSON data.")
    
    logger.info(f"Loaded {len(df)} policy documents from {filename}")
    return df

def save_similarity_matrix(matrix, index_labels, filename="similarity_matrix.json"):
    """
    Saves the pairwise similarity matrix as a JSON file.
    This function stores data as a list of { "doc1": ..., "doc2": ..., "similarity": ... } records,
    which is more convenient for many analyses than a raw NxN matrix.
    """
    logger.info(f"Saving similarity matrix to {filename}")
    results = []
    n = len(index_labels)
    for i in range(n):
        for j in range(i+1, n):
            results.append({
                "doc1": index_labels[i],
                "doc2": index_labels[j],
                "similarity": matrix[i][j]
            })
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved pairwise similarities: {len(results)} total pairs")

###############################################################################
# 3. PROMPT ENGINEERING & CHUNKING
###############################################################################
def build_embedding_prompt(text_chunk):
    """
    Creates a prompt that instructs Ollama to return ONLY a space-separated list of floats
    representing the embedding for text_chunk.
    """
    return f"""
You are a specialized embedding model. Given the following text:
----
{text_chunk}
----
Generate a single-line embedding: a sequence of floating point values (space-separated).
No additional text or explanation, only the embedding line.
""".strip()

def chunk_text(text, max_length=1000):
    """
    Splits large text into smaller chunks of ~max_length characters each.
    This helps avoid overly long prompts that might exceed model or service limits.
    """
    chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]
    return chunks

###############################################################################
# 4. OLLAMA CALLS & EMBEDDING LOGIC
###############################################################################
def parse_embedding_response(response_text):
    """
    Parses Ollama's raw embedding output (e.g., "0.123 0.456 -0.789") into a Python list of floats.
    """
    try:
        cleaned = response_text.strip()
        values = cleaned.split()
        embedding = [float(v) for v in values]
        return embedding
    except Exception as e:
        logger.error(f"Error parsing embedding: {e} | Raw text: {response_text}")
        return None

def call_ollama_for_chunk_embedding(text_chunk):
    """
    Calls Ollama to get the embedding for one chunk of text.
    Returns a Python list of floats if successful, else None.
    """
    prompt = build_embedding_prompt(text_chunk)
    try:
        response = ollama.generate(model="llama3.1", prompt=prompt)
        raw_text = response.get("response", "")
        return parse_embedding_response(raw_text)
    except Exception as e:
        logger.error(f"Ollama error while embedding chunk: {e}")
        return None

def generate_document_embedding(text, chunk_size=1000):
    """
    Splits the text into chunks (to handle large policy docs),
    generates an embedding for each chunk, and averages them
    to produce a single embedding vector per document.
    """
    # Split text into manageable chunks
    text_chunks = chunk_text(text, max_length=chunk_size)
    
    chunk_embeddings = []
    for chunk in text_chunks:
        emb = call_ollama_for_chunk_embedding(chunk)
        if emb is not None:
            chunk_embeddings.append(emb)
        else:
            logger.warning("Failed to retrieve embedding for a chunk. Skipping that chunk.")
    
    if not chunk_embeddings:
        return None
    
    # Average all chunk embeddings to form one embedding for the entire document
    emb_array = np.array(chunk_embeddings)
    avg_emb = np.mean(emb_array, axis=0)
    return avg_emb.tolist()

###############################################################################
# 5. COSINE SIMILARITY
###############################################################################
def compute_cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two equal-length embeddings.
    Returns None if there's a mismatch or zero-length vector.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if v1.shape != v2.shape:
        logger.error("Embedding shape mismatch for cosine similarity.")
        return None
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        logger.warning("Zero-norm vector encountered. Similarity set to None.")
        return None
    
    return float(np.dot(v1, v2) / (norm1 * norm2))

###############################################################################
# 6. GENERATE EMBEDDINGS + PAIRWISE SIMILARITIES
###############################################################################
def generate_embeddings_for_all(df, text_col="policy_text", concurrency=4, chunk_size=1000):
    """
    Embeds each policy document in the DataFrame.
    Utilizes a ThreadPoolExecutor for concurrency.
    The resulting embeddings are stored in a new column "embedding".
    """
    logger.info(f"Generating embeddings for {len(df)} documents (concurrency={concurrency})")
    
    def embed_single_document(idx, policy_text):
        return (idx, generate_document_embedding(policy_text, chunk_size))
    
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {}
        
        for idx, row in df.iterrows():
            policy_text = row[text_col]
            future = executor.submit(embed_single_document, idx, policy_text)
            future_to_idx[future] = idx
        
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Embedding"):
            idx = future_to_idx[future]
            try:
                i, embedding = future.result()
                results.append((i, embedding))
            except Exception as e:
                logger.error(f"Exception generating embedding for row {idx}: {e}")
                results.append((idx, None))
    
    # Create a mapping from idx -> embedding
    idx_to_emb = dict(results)
    
    # Insert embeddings into the DataFrame
    emb_list = [idx_to_emb.get(i) for i in df.index]
    df["embedding"] = emb_list
    return df

def compute_pairwise_similarities(df, id_col="id"):
    """
    Computes pairwise cosine similarity for every document in df based on the "embedding" column.
    Returns:
      - similarity_matrix (NxN numpy array) 
      - labels (list) parallel to the matrix's dimensions, typically the doc IDs
    """
    # Get embeddings in the same order
    embeddings = df["embedding"].tolist()
    if id_col in df.columns:
        labels = df[id_col].tolist()
    else:
        # Fallback if ID is not present, just use index
        labels = [str(i) for i in df.index]
    
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n), dtype=float)
    
    logger.info(f"Computing pairwise similarities for {n} documents...")
    for i in tqdm(range(n), desc="Similarity"):
        for j in range(i+1, n):
            if embeddings[i] is not None and embeddings[j] is not None:
                sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                # If sim is None, treat as 0.0
                sim_val = sim if sim is not None else 0.0
            else:
                sim_val = 0.0
            
            similarity_matrix[i, j] = sim_val
            similarity_matrix[j, i] = sim_val
    
    return similarity_matrix, labels

###############################################################################
# 7. DISPLAY THE SIMILARITY MATRIX
###############################################################################
def display_similarity_matrix(matrix, labels, max_rows=8, decimal_places=3):
    """
    Prints the top-left corner of the similarity matrix in a tabular format.
    
    Args:
        matrix (np.array): NxN similarity matrix.
        labels (list): A list of length N with document identifiers (strings).
        max_rows (int): Max number of rows/columns to display (for large N, we truncate).
        decimal_places (int): Number of decimal places to show in each cell.
    """
    n = len(labels)
    rows_to_print = min(max_rows, n)
    
    print("\n=== Pairwise Similarity Matrix (Top-Left) ===")
    
    # Print header row (truncated labels)
    print(" " * 12, end="")
    for col_idx in range(rows_to_print):
        print(f"{labels[col_idx][:8]:>10}", end=" ")
    print()
    
    # Print each row
    for i in range(rows_to_print):
        row_label = labels[i][:8]  # truncated label
        print(f"{row_label:>10}", end=" ")
        for j in range(rows_to_print):
            val = matrix[i][j]
            if val is not None:
                print(f"{val:.{decimal_places}f}".rjust(10), end=" ")
            else:
                print(f"{'None':>10}", end=" ")
        print()
    
    if rows_to_print < n:
        print(f"... (Matrix truncated to {max_rows}x{max_rows})\n")

def filter_rows_by_nonzero_threshold(sim_matrix, labels, fraction=0.1):
    """
    Removes rows and columns (and corresponding labels) from 'sim_matrix'
    where the row has nonzero similarity with fewer than 'fraction' of the columns.
    
    Args:
        sim_matrix (np.array): NxN similarity matrix.
        labels (list): Corresponding labels of length N.
        fraction (float): The fraction of columns that must have nonzero similarity.
                          e.g., 0.5 means at least half must be nonzero.
    
    Returns:
        filtered_matrix (np.array): The filtered NxN similarity matrix.
        filtered_labels (list): The filtered labels.
    """
    # Count non-zero similarities for each row
    nonzero_counts = np.count_nonzero(sim_matrix, axis=1)
    
    # Calculate threshold based on 'fraction'
    threshold = sim_matrix.shape[1] * fraction
    
    # Create a boolean mask for rows that meet this requirement
    keep_mask = nonzero_counts >= threshold
    
    # Filter matrix (both rows and columns) and labels
    filtered_matrix = sim_matrix[keep_mask][:, keep_mask]
    filtered_labels = [label for keep, label in zip(keep_mask, labels) if keep]
    
    return filtered_matrix, filtered_labels

def plot_similarity_matrix(matrix, labels, title="Similarity Matrix", figsize=(5, 5), cmap="magma"):
    """
    Plots a similarity matrix as a heatmap with labeled axes and colorbar.
    Makes the plot more condensed by reducing figure size, shrinking color bar,
    and adjusting font sizes and spacing.
    
    Args:
        matrix (np.array): NxN similarity matrix.
        labels (list): List of document labels for rows/columns of the matrix.
        title (str): Title of the heatmap.
        figsize (tuple): Size of the figure.
        cmap (str): Colormap for the heatmap (e.g., "magma", "coolwarm").
    """

    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap="magma",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.5},
        square=True,
        vmin=0.0,
        vmax=1.0
    )
    plt.figure(figsize=(5,5))


    # Title and axes with smaller font sizes
    plt.title(title, fontsize=10, pad=5)
    plt.xlabel("Policies", fontsize=8)
    plt.ylabel("Policies", fontsize=8)

    # Tight layout with smaller padding
    plt.tight_layout(pad=0.5)

    plt.show()
###############################################################################
# 8. MAIN ENTRY POINT
###############################################################################
if __name__ == "__main__":
    """    # 1) LOAD DATA
    input_file = "all_policies.json"  # The single file with all policy texts
    df = load_dataset(filename=input_file, text_col="policy_text")

    # 2) GENERATE EMBEDDINGS FOR EACH POLICY
    df = generate_embeddings_for_all(
        df, 
        text_col="policy_text", 
        concurrency=8,   # Adjust threads as appropriate
        chunk_size=12000  # Tweak chunk size for your typical doc lengths
    )

    # 3) COMPUTE PAIRWISE COSINE SIMILARITIES
    sim_matrix, labels = compute_pairwise_similarities(df, id_col="id")

    # 4) SAVE RESULTS
    output_file = "policy_similarity_results.json"
    save_similarity_matrix(sim_matrix, labels, filename=output_file)

    # 5) DISPLAY A PORTION OF THE SIMILARITY MATRIX IN THE CONSOLE
    display_similarity_matrix(sim_matrix, labels, max_rows=8, decimal_places=3)

    logger.info("Done! Check logs and output files for results.")"""
    
    # 1) Load the precomputed similarity matrix
    similarity_file = "policy_similarity_results.json"  # JSON file with precomputed results

    # 2) Load similarity matrix and labels
    with open(similarity_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract labels and rebuild the similarity matrix
    labels = list({pair["doc1"] for pair in data} | {pair["doc2"] for pair in data})  # Unique labels
    labels.sort()  # Ensure consistent order
    n = len(labels)
    
    # Create an NxN matrix initialized to zeros
    sim_matrix = np.zeros((n, n), dtype=float)
    label_to_index = {label: idx for idx, label in enumerate(labels)}  # Map labels to indices

    # Populate the similarity matrix
    for pair in data:
        i, j = label_to_index[pair["doc1"]], label_to_index[pair["doc2"]]
        sim_matrix[i, j] = pair["similarity"]
        sim_matrix[j, i] = pair["similarity"]  # Symmetric

    sim_matrix, labels = filter_rows_by_nonzero_threshold(sim_matrix, labels)
    # 3) Plot the similarity matrix as a heatmap
    plot_similarity_matrix(
        sim_matrix,
        labels,
        title="Pairwise Document Similarity Heatmap (Loaded Data)",
        figsize=(12, 10),
        cmap="coolwarm"
    )
