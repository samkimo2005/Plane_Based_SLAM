import sys

import numpy as np
import scipy
import cv2
import open3d as o3d
import matplotlib


print("=== PlaneBasedSLAM Environment Test ===")

print(f"Python:      {sys.version}")
print(f"NumPy:       {np.__version__}")
print(f"SciPy:       {scipy.__version__}")
print(f"OpenCV:      {cv2.__version__}")
print(f"Open3D:      {o3d.__version__}")
print(f"Matplotlib:  {matplotlib.__version__}")

print("\nAll core Python libraries imported successfully.")

# Simple NumPy test
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

dot_product = np.dot(a, b)

print(f"\nNumPy dot-product test: {dot_product}")

# Simple OpenCV test
image = np.zeros((200, 300, 3), dtype=np.uint8)

print(f"OpenCV image test shape: {image.shape}")

# Simple Open3D test
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5,
    origin=[0, 0, 0]
)

print("\nOpening Open3D test window...")
print("Close the Open3D window to finish the test.")

o3d.visualization.draw_geometries([coordinate_frame])

print("\n=== Environment Test PASSED ===")