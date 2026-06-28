import tifffile
import numpy as np

class MockDatasetReader:
    def __init__(self, path):
        self.path = path
        self.image = tifffile.imread(path)
        # Handle tifffile shape (it may be H, W, C or C, H, W)
        if len(self.image.shape) == 2:
            self.image = np.expand_dims(self.image, axis=0) # (1, H, W)
        elif len(self.image.shape) == 3:
            if self.image.shape[-1] <= 10: # Likely (H, W, C)
                self.image = np.transpose(self.image, (2, 0, 1))
        
        self.count = self.image.shape[0]
        self.meta = {
            'width': self.image.shape[2],
            'height': self.image.shape[1],
            'crs': 'EPSG:4326',
            'transform': None,
            'count': self.count,
            'dtype': str(self.image.dtype)
        }

    def read(self):
        return self.image

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def open(path, *args, **kwargs):
    return MockDatasetReader(path)
