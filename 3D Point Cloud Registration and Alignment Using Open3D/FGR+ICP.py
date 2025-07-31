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

# Adaptive voxel downsampling and feature extraction
def adaptive_voxel_size(pcd, target_size=22):
    bbox = pcd.get_axis_aligned_bounding_box()
    size = np.linalg.norm(np.array(bbox.get_max_bound()) - np.array(bbox.get_min_bound()))
    voxel_size = min(target_size, size / 50)
    return voxel_size

# Fast Global Registration (FGR) function
def perform_fast_global_registration(source_down, target_down, voxel_size, distance_threshold=0.05):
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

# ICP refinement function
def perform_icp_refinement(pcd_a, pcd_b, init_transformation, voxel_size):
    pcd_b.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    icp_result = o3d.pipelines.registration.registration_icp(
        pcd_a, pcd_b,
        max_correspondence_distance=voxel_size * 1.2,
        init=init_transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
    return icp_result

# RMSE calculation between projected points and image reference
def calculate_rmse_image(proj_points_2d, img_width, img_height, shift_x, shift_y):
    # Here we compute distances from projected points to a reference (shift_x, shift_y)
    distances = np.sqrt((proj_points_2d[:, 0] - shift_x)**2 + (proj_points_2d[:, 1] - shift_y)**2)
    rmse_image = np.sqrt(np.mean(distances**2))
    max_error = np.sqrt((img_width - shift_x)**2 + (img_height - shift_y)**2)
    rmse_percentage = (rmse_image / max_error) * 100
    return rmse_image, rmse_percentage

# Load the .pcd files
pcd_a = o3d.io.read_point_cloud("/content/a.pcd")
pcd_b = o3d.io.read_point_cloud("/content/b.pcd")

# Preprocess point clouds with adaptive voxel sizes
voxel_size_a = adaptive_voxel_size(pcd_a)
voxel_size_b = adaptive_voxel_size(pcd_b)
pcd_a_down = preprocess_point_cloud(pcd_a, voxel_size_a)
pcd_b_down = preprocess_point_cloud(pcd_b, voxel_size_b)

# Perform Fast Global Registration (FGR)
start_time = time.time()
result_fgr = perform_fast_global_registration(pcd_a_down, pcd_b_down, voxel_size_a, distance_threshold=voxel_size_a)
print(f"Fast Global Registration completed in {(time.time() - start_time):.2f} seconds.")

# Perform ICP refinement with Point-to-Plane method
start_time = time.time()
icp_result = perform_icp_refinement(pcd_a, pcd_b, result_fgr.transformation, voxel_size_a)
print(f"ICP refinement completed in {(time.time() - start_time):.2f} seconds.")

# Now, transform b to align with a using ICP result
pcd_b.transform(icp_result.transformation)

# Combine the aligned point clouds (pcd_a and pcd_b) into a single point cloud
combined_points = np.concatenate((np.asarray(pcd_a.points), np.asarray(pcd_b.points)), axis=0)

# Project the combined 3D points to 2D (using X and Y coordinates for simple 2D projection)
combined_points_2d = combined_points[:, :2]

# Adjust the scaling and shifting values
scale_factor = 0.25
shift_x, shift_y = 200, 200

# Adjust the points to the image space (scaling and shifting)
combined_points_2d = combined_points_2d * scale_factor + [shift_x, shift_y]

# Load the target image (a_b.png)
img_path = "/content/a_b.png"
img = mpimg.imread(img_path)

# Create the plot to overlay the points on the image
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.scatter(combined_points_2d[:, 0], combined_points_2d[:, 1], s=1, c='blue', label="Aligned Point Cloud")
plt.legend()
plt.show()

# Calculate and print the RMSE between the projected points and the target image
rmse_image, rmse_percentage = calculate_rmse_image(combined_points_2d, img.shape[1], img.shape[0], shift_x, shift_y)
print(f"RMSE as percentage of the maximum possible error between the aligned points and the image: {rmse_percentage:.2f}%")
