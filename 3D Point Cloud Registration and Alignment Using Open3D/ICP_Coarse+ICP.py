!pip install open3d
import open3d as o3d
import numpy as np
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Function to preprocess point cloud (adaptive voxel downsampling and normal estimation)
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

# Adaptive voxel downsampling
def adaptive_voxel_size(pcd, target_size=5):
    bbox = pcd.get_axis_aligned_bounding_box()
    size = np.linalg.norm(np.array(bbox.get_max_bound()) - np.array(bbox.get_min_bound()))
    voxel_size = min(target_size, size / 50)
    return voxel_size

# ICP for coarse registration (replacing NDT due to AttributeError)
def perform_icp_coarse(source_down, target_down, voxel_size, max_correspondence_distance=0.5):
    icp_result_coarse = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        max_correspondence_distance,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return icp_result_coarse

# ICP refinement function
def perform_icp_refinement(pcd_a, pcd_b, init_transformation, voxel_size):
    pcd_b.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=1000)
    )
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_a, pcd_b,
        max_correspondence_distance=voxel_size * 1.2,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
    return icp_result

# RMSE calculation between projected points and image reference with timing
def calculate_rmse_image(proj_points_2d, img_width, img_height, shift_x, shift_y):
    start = time.time()
    distances = np.sqrt((proj_points_2d[:, 0] - shift_x)**2 + (proj_points_2d[:, 1] - shift_y)**2)
    rmse_image = np.sqrt(np.mean(distances**2))
    max_error = np.sqrt((img_width - shift_x)**2 + (img_height - shift_y)**2)
    rmse_percentage = (rmse_image / max_error) * 100
    end = time.time()
    rmse_time = end - start
    return rmse_image, rmse_percentage, rmse_time

# Load the .pcd files
pcd_a = o3d.io.read_point_cloud("/content/a.pcd")
pcd_b = o3d.io.read_point_cloud("/content/b.pcd")

# Preprocess point clouds with adaptive voxel sizes
voxel_size_a = adaptive_voxel_size(pcd_a)
voxel_size_b = adaptive_voxel_size(pcd_b)
pcd_a_down = preprocess_point_cloud(pcd_a, voxel_size_a)
pcd_b_down = preprocess_point_cloud(pcd_b, voxel_size_b)

# Perform ICP coarse registration timing only
start_time = time.time()
result_coarse_icp = perform_icp_coarse(pcd_a_down, pcd_b_down, voxel_size_a, max_correspondence_distance=voxel_size_a * 1.5)
icp_coarse_time = time.time() - start_time

# Perform ICP refinement timing only
start_time = time.time()
icp_result = perform_icp_refinement(pcd_a, pcd_b, result_coarse_icp.transformation, voxel_size_a)
icp_refine_time = time.time() - start_time

# Apply ICP refinement transformation to pcd_b
pcd_b.transform(icp_result.transformation)

# Combine aligned point clouds
combined_points = np.concatenate((np.asarray(pcd_a.points), np.asarray(pcd_b.points)), axis=0)

# Project 3D points to 2D
combined_points_2d = combined_points[:, :2]

# Scale and shift for image space
scale_factor = 0.25
shift_x, shift_y = 200, 200
combined_points_2d = combined_points_2d * scale_factor + [shift_x, shift_y]

# Load target image
img_path = "/content/a_b.png"
img = mpimg.imread(img_path)

# Visualize overlay
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.scatter(combined_points_2d[:, 0], combined_points_2d[:, 1], s=1, c='blue', label="Aligned Point Cloud")
plt.legend()
plt.show()

# Calculate RMSE and timing
rmse_image, rmse_percentage, rmse_time = calculate_rmse_image(combined_points_2d, img.shape[1], img.shape[0], shift_x, shift_y)

# Print only the requested info
print(f"ICP coarse registration time: {icp_coarse_time:.3f} seconds")
print(f"ICP refinement time: {icp_refine_time:.3f} seconds")

print(f"RMSE as percentage of maximum possible error: {rmse_percentage:.2f}%")
