import numpy as np
from transforms.spatial import SpatialTransforms
from transforms.geometric import GeometricTransforms

def run():
    print("[INFO] Initializing AugmentaCV Pipeline...")
    dummy_image = np.full((300, 300, 3), 128, dtype=np.uint8)

    # Apply spatial and geometric operations
    _ = SpatialTransforms.brightness(dummy_image, beta=40)
    _ = SpatialTransforms.contrast(dummy_image, alpha=1.4)
    _ = GeometricTransforms.rotate(dummy_image, angle=45)

    print("[SUCCESS] All pipeline transformations executed successfully.")

if __name__ == "__main__":
    run()
