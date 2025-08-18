import numpy as np

def convert_numpy_to_python(data):
    """
    Recursively traverses a dictionary or list and converts numpy data types
    to their native Python equivalents.
    """
    if isinstance(data, dict):
        return {k: convert_numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float64, np.float32, np.float16)):
        return float(data)
    elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    # Add other numpy types if needed
    return data