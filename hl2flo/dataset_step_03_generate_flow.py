#------------------------------------------------------------------------------
# Step 03: Generate optical flow.
#------------------------------------------------------------------------------

import numpy as np
import os
import cv2
import flow_viz
import hl2ss
import hl2ss_3dcv
import hl2flo

# Settings --------------------------------------------------------------------

# Data folder
path = './data/test_data/'

# Flow stride
# Compute flow from frame n to frame n + stride
stride = 1

# Fill value
invalid_set = 0

#------------------------------------------------------------------------------

# Data Load -------------------------------------------------------------------
pv_in = os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)}')
lt_in = os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)}')
wp_in = os.path.join(path, 'pose')

color_intrinsics = np.fromfile(os.path.join(path, 'pv_intrinsics.bin'), dtype=np.float32).reshape((4, 4))
color_extrinsics = np.fromfile(os.path.join(path, 'pv_extrinsics.bin'), dtype=np.float32).reshape((4, 4))

print('Camera intrinsics')
print(color_intrinsics)
print('Camera extrinsics')
print(color_extrinsics)

files_pv = hl2flo.scan_files(pv_in)
files_lt = hl2flo.scan_files(lt_in)
files_wp = hl2flo.scan_files(wp_in)

files = zip(files_pv[:-stride], files_lt[:-stride], files_wp[:-stride], files_pv[stride:], files_wp[stride:])

pv_image = cv2.imread(files_pv[0])

pv_height, pv_width, pv_channels = pv_image.shape

depth_scale = 1000
uv2xy       = hl2ss_3dcv.compute_uv2xy(color_intrinsics, pv_width, pv_height)
xy1, scale  = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, depth_scale)

xy = np.dstack(np.meshgrid(np.arange(pv_width, dtype=np.float32), np.arange(pv_height, dtype=np.float32)))

# Create output folder --------------------------------------------------------
flo_out     = os.path.join(path, f'flow_{stride}')
flo_vis_out = os.path.join(path, f'flow_vis_{stride}')
warp_out    = os.path.join(path, f'flow_warp_{stride}')

os.makedirs(flo_out)
os.makedirs(flo_vis_out)
os.makedirs(warp_out)

# Main loop -------------------------------------------------------------------
index = 0

for file_pv0, file_lt, file_wp0, file_pv1, file_wp1 in files:
    # Load data ---------------------------------------------------------------
    pv_rgb0  = cv2.imread(file_pv0)
    pv_z     = cv2.imread(file_lt, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
    pv_pose0 = np.fromfile(file_wp0, dtype=np.float32).reshape((4, 4))
    pv_rgb1  = cv2.imread(file_pv1)
    pv_pose1 = np.fromfile(file_wp1, dtype=np.float32).reshape((4, 4))
    pv_mask0 = pv_z <= 0

    # Project 3D points from image 0 into image1 to obtain 2D correspondences -
    pv_points0   = hl2ss_3dcv.slice_to_block(pv_z) * xy1
    pv_to_world0 = hl2ss_3dcv.camera_to_rignode(color_extrinsics) @ hl2ss_3dcv.reference_to_world(pv_pose0)
    world_to_pv1 = hl2ss_3dcv.world_to_reference(pv_pose1) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics)
    pv_points1   = hl2ss_3dcv.transform(pv_points0, pv_to_world0 @ world_to_pv1)
    pv_pixels1   = hl2ss_3dcv.project(pv_points1, hl2ss_3dcv.camera_to_image(color_intrinsics))

    # Warp image 1 using 2D correspondences for qualitative evaluation --------
    # Warped image 1 should be similar to image 0
    pv_pixels1[np.dstack((pv_mask0, pv_mask0))] = -1
    pv_rgb01 = cv2.remap(pv_rgb1, pv_pixels1[:, :, 0], pv_pixels1[:, :, 1], cv2.INTER_LINEAR)

    # Compure optical flow as the difference of the 2D correspondences --------
    flow1 = pv_pixels1 - xy
    flow1[np.dstack((pv_mask0, pv_mask0))] = invalid_set

    flow1_vis = flow_viz.flow_to_image(flow1, convert_to_bgr=True)

    # Write data --------------------------------------------------------------
    hl2flo.flow_to_flo(flow1, os.path.join(flo_out, f'{index:06d}.flo'))
    cv2.imwrite(os.path.join(flo_vis_out, f'{index:06d}.png'), flow1_vis)
    cv2.imwrite(os.path.join(warp_out, f'{index:06d}.png'), pv_rgb01)

    index += 1

    # Display results ---------------------------------------------------------
    cv2.imshow('RGB Warp', np.vstack((np.hstack((pv_rgb0, pv_rgb01)), np.hstack((pv_rgb1, flow1_vis)))))
    cv2.waitKey(1)

# End -------------------------------------------------------------------------
print(f'Flows saved to {flo_out}')
print(f'Flows (color) saved to {flo_vis_out}')
print(f'Warped images saved to {warp_out}')
