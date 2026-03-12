"""
Hybrid Search Implementation
Combines Cosine Similarity (Embedding) + Normalized DTW (Raw Series)
Optimized with fast DTW approximation and caching.
"""
import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class HybridRetriever:
    def __init__(self, dtw_window_size=None, fast_dtw_max_len=100):
        self.ids = []
        self.labels = []
        self.vectors = []     # List of numpy arrays [Dim]
        self.ts_data = []     # List of numpy arrays [Time] or [Dim, Time]
        self._built = False
        self._vector_matrix = None
        self.dtw_window_size = dtw_window_size
        self.fast_dtw_max_len = int(fast_dtw_max_len)

    def add(self, item_id, embedding, time_series, label=None):
        """
        Add a single item to the index.
        item_id: str/int
        embedding: numpy array [D]
        time_series: numpy array [T] or [D, T]
        """
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        if isinstance(time_series, list):
            time_series = np.array(time_series)
            
        self.ids.append(item_id)
        self.labels.append(None if label is None else str(label))
        self.vectors.append(embedding)
        self.ts_data.append(time_series)
        self._built = False

    def build_index(self):
        """
        Prepare vectors for fast cosine similarity.
        """
        if not self.vectors:
            print("Warning: No vectors to build index.")
            return
            
        # Stack and Normalize
        matrix = np.array(self.vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._vector_matrix = matrix / (norms + 1e-10)
        self._built = True
        print(f"Index built with {len(self.ids)} items.")


    def _compute_dtw(self, s1, s2, window_size=None):
        """
        Optimized DTW implementation with Sakoe-Chiba band constraint.
        s1, s2: [Time, Channels] or [Channels, Time]
        window_size: Sakoe-Chiba band width (max warping distance)
        """
        # Ensure format [Time, Channels] for easier distance computation
        if s1.ndim == 1:
            s1 = s1.reshape(-1, 1)
        if s2.ndim == 1:
            s2 = s2.reshape(-1, 1)

        if s1.shape[0] < s1.shape[1]:
             s1 = s1.T  # [T, C]
        if s2.shape[0] < s2.shape[1]:
             s2 = s2.T  # [T, C]

        n, m = s1.shape[0], s2.shape[0]

        # Set window size (Sakoe-Chiba band)
        if window_size is None:
            window_size = self.dtw_window_size
        if window_size is None:
            window_size = max(n, m) // 4  # 25% warping constraint

        # Precompute distance matrix using vectorized operations
        # Use cdist for efficient pairwise distance computation
        dist_matrix = cdist(s1, s2, metric='euclidean')

        # Initialize DP matrix with infinity
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0

        # Fill DP matrix with Sakoe-Chiba band constraint
        for i in range(1, n + 1):
            # Calculate valid j range based on window
            j_start = max(1, i - window_size)
            j_end = min(m, i + window_size) + 1

            for j in range(j_start, j_end):
                cost = dist_matrix[i-1, j-1]
                dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                       dtw[i, j-1],    # deletion
                                       dtw[i-1, j-1])  # match

        return dtw[n, m]

    def _compute_fast_dtw(self, s1, s2):
        """
        Fast DTW approximation using Euclidean distance on downsampled sequences.
        Good for long time series.
        """
        # Downsample if sequences are too long
        max_len = max(1, int(self.fast_dtw_max_len))
        if len(s1) > max_len or len(s2) > max_len:
            # Simple downsampling
            stride1 = max(1, len(s1) // max_len)
            stride2 = max(1, len(s2) // max_len)
            s1_ds = s1[::stride1]
            s2_ds = s2[::stride2]
            downsampled_window = self.dtw_window_size
            if downsampled_window is None:
                downsampled_window = min(20, max(s1_ds.shape[0], s2_ds.shape[0]))
            return self._compute_dtw(s1_ds, s2_ds, window_size=downsampled_window)
        else:
            return self._compute_dtw(s1, s2)

    def search(self, query_vec, query_ts, k=5, alpha=0.7):
        """
        Hybrid Search: alpha * Semantic + (1-alpha) * Structural
        query_vec: [D]
        query_ts: [T] or [D, T] matches stored format
        alpha: weight for Semantic Score (0.0 to 1.0)
        """
        if not self._built:
            self.build_index()

        # 1. Semantic Score (Cosine Similarity)
        # Normalize Query
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        # Dot product
        # [N, D] @ [D] -> [N]
        cosine_sims = np.dot(self._vector_matrix, q_norm)
        # Map Cosine [-1, 1] -> [0, 1] roughly for fusion
        sem_scores = (cosine_sims + 1) / 2.0

        # 2. Structural Score (DTW)
        # Filter top-M candidates by semantic score to speed up DTW calculation
        M = min(len(self.ids), max(k * 10, 100))
        top_m_indices = np.argsort(sem_scores)[::-1][:M]

        struct_scores = np.zeros(len(self.ids))
        
        # Z-Normalize query once
        q_mean = np.mean(query_ts)
        q_std = np.std(query_ts) + 1e-10
        q_norm_ts = (query_ts - q_mean) / q_std
        
        for idx in top_m_indices:
            ref_ts = self.ts_data[idx]
            # Z-Normalize ref
            r_mean = np.mean(ref_ts)
            r_std = np.std(ref_ts) + 1e-10
            r_norm_ts = (ref_ts - r_mean) / r_std
            
            # Calculate DTW distance on NORMALIZED series
            # Use fast DTW for efficiency
            dist = self._compute_fast_dtw(q_norm_ts, r_norm_ts)
            
            # Convert distance to similarity [0, 1]
            sim = 1.0 / (1.0 + dist)
            struct_scores[idx] = sim
            
        # Normalize structural scores to be competitive with semantic scores [0,1]
        evaluated_scores = struct_scores[top_m_indices]
        if evaluated_scores.max() > evaluated_scores.min():
             # Min-Max Normalization
             evaluated_norm = (evaluated_scores - evaluated_scores.min()) / (evaluated_scores.max() - evaluated_scores.min())
             struct_scores[top_m_indices] = evaluated_norm

        # 3. Fusion
        final_scores = alpha * sem_scores + (1 - alpha) * struct_scores

        # 4. Top K
        best_of_m = top_m_indices[np.argsort(final_scores[top_m_indices])[::-1][:k]]
        
        results = []
        for idx in best_of_m:
            results.append({
                "id": self.ids[idx],
                "label": self.labels[idx] if idx < len(self.labels) else None,
                "score": float(final_scores[idx]),
                "sem_score": float(sem_scores[idx]),
                "struct_score": float(struct_scores[idx]),
                "raw_ts_sample": self.ts_data[idx].flatten()[:10].tolist() # Preview
            })
            
        return results

    def batch_search(self, query_vecs, query_ts_list, k=5, alpha=0.7):
        """
        Batch search for multiple queries.
        query_vecs: list of numpy arrays [D]
        query_ts_list: list of numpy arrays [T] or [D, T]
        """
        results = []
        for query_vec, query_ts in zip(query_vecs, query_ts_list):
            result = self.search(query_vec, query_ts, k=k, alpha=alpha)
            results.append(result)
        return results

    def save_index(self, filepath):
        """
        Save the index to disk.
        """
        import pickle
        data = {
            'ids': self.ids,
            'labels': self.labels,
            'vectors': self.vectors,
            'ts_data': self.ts_data,
            '_built': self._built,
            '_vector_matrix': self._vector_matrix
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath):
        """
        Load the index from disk.
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.ids = data['ids']
        self.labels = data.get('labels', [None] * len(self.ids))
        self.vectors = data['vectors']
        self.ts_data = data['ts_data']
        self._built = data['_built']
        self._vector_matrix = data['_vector_matrix']
        print(f"Index loaded from {filepath} with {len(self.ids)} items")
