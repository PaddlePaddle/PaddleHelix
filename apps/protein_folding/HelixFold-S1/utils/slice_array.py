import numpy as np

def slice_array(array, start, length, dim):
    """
    Slice a NumPy array on a specified dimension.
    
    Parameters:
    array (np.ndarray): The NumPy array to be sliced.
    start (int): The starting index for the slice.
    length (int): The length of the slice.
    dim (int): The dimension along which to slice. (0-indexed)
    
    Returns:
    np.ndarray: The sliced NumPy array.
    """
    # Get the shape of the input array
    shape = array.shape
    
    # Check if the specified dimension is within the valid range
    if dim < 0 or dim >= len(shape):
        raise ValueError(f"Dimension index {dim} is out of range for array with shape {shape}")
    
    # Check if the slice start and length are within the bounds of the specified dimension
    if start < 0 or start + length > shape[dim]:
        raise ValueError(f"Slice start:{start} and length:{length} is out of bounds for dimension {dim} with size {shape[dim]}")
    
    # Create a slice object for the specified dimension
    slice_obj = slice(start, start + length)
    
    # Build an indexing tuple using slice_obj for the specified dimension and slice(None) for all other dimensions
    # slice(None) is equivalent to selecting all elements in that dimension
    index_tuple = tuple(slice_obj if i == dim else slice(None) for i in range(len(shape)))
    
    # Use the indexing tuple to slice the array
    sliced_array = array[index_tuple]
    
    # Return the sliced array
    return sliced_array


if __name__ == "__main__":
    # Example usage
    A = np.random.rand(4, 5, 6)  # Create a random array of shape (4, 5, 6)
    try:
        sliced = slice_array(A, start=1, length=3, dim=1)  # Slice the array on the second dimension (index 1) from index 1 with a length of 3
        print("Shape of the sliced array:", sliced.shape)  # Output should be (4, 3, 6)
    except ValueError as e:
        print("Error:", e)  # Catch and print any value errors that may occur due to invalid inputs