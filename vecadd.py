import numpy as np

#Adding this command to mark my change that i made on old code in branch master
#now here i have resolvd the merge conflict

def add_vectors(vec1 , vec2):
  """
  Performs element-wise addition of two vectors.

from multiprocessing import Pool

def partial_dot_product(args):
    """Calculates the dot product for a segment of the vectors."""
    vec_a_segment, vec_b_segment = args
    return np.dot(vec_a_segment, vec_b_segment)


def parallel_dot_product(vec_a, vec_b, num_processes=None):
    """
    Computes the dot product of two vectors in parallel using multiprocessing.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length.")

    if num_processes is None:
        num_processes = Pool()._processes # Use default number of processes

    segment_size = len(vec_a) // num_processes
    segments_a = [vec_a[i * segment_size : (i + 1) * segment_size] for i in range(num_processes - 1)]
    segments_a.append(vec_a[(num_processes - 1) * segment_size :]) # Handle remaining elements

    segments_b = [vec_b[i * segment_size : (i + 1) * segment_size] for i in range(num_processes - 1)]
    segments_b.append(vec_b[(num_processes - 1) * segment_size :])

    with Pool(num_processes) as pool:
        partial_results = pool.map(partial_dot_product, zip(segments_a, segments_b))

    return sum(partial_results)

# Example usage:
if __name__ == "__main__":
    vector1 = np.random.rand(1000000)
    vector2 = np.random.rand(1000000)

    # Parallel computation
    result_parallel = parallel_dot_product(vector1, vector2, num_processes=4)
    print(f"Parallel dot product: {result_parallel}")

    # Standard NumPy computation for comparison
    result_numpy = np.dot(vector1, vector2)
    print(f"NumPy dot product: {result_numpy}")
