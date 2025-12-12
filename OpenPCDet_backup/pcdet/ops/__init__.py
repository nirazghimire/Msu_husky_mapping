import warnings
warnings.warn("Running OpenPCDet in CPU-only mode: CUDA ops are disabled.")

# Create dummy fallback imports for CPU mode
from .iou3d_nms import iou3d_nms_utils
from .roiaware_pool3d import roiaware_pool3d_utils
try:
    from .pointnet2 import pointnet2_utils
except ImportError:
    pointnet2_utils = None
    print("Warning: pointnet2_utils not found")

# Monkey patch functions that depend on CUDA
iou3d_nms_utils.nms_gpu = None
roiaware_pool3d_utils.roiaware_pool3d_gpu = None
if pointnet2_utils is not None:
    pointnet2_utils.PointNetSetAbstraction = None

