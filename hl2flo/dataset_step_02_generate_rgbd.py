#------------------------------------------------------------------------------
# Step 02: Generate aligned RGB-D pairs.
#------------------------------------------------------------------------------

import numpy as np
import os
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_io
import hl2ss_mx
import hl2ss_3dcv

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.1.7'

# Calibration folder
# Must exist
# If empty, calibration will be downloaded from the HoloLens
calibration_path = '../calibration'

# Data folder
path = './data/HL2-2025-05-13-02-38-33/'

#------------------------------------------------------------------------------

# Create output folders -------------------------------------------------------
path_out_pv = os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)}')
path_out_lt = os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)}')
path_out_wp = os.path.join(path, 'pose')
path_out_vz = os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)}_colormap')

os.makedirs(path_out_pv)
os.makedirs(path_out_lt)
os.makedirs(path_out_wp)
os.makedirs(path_out_vz)

# Get RM Depth Long Throw calibration -----------------------------------------
# Calibration data will be downloaded if it's not in the calibration folder
calibration_lt = hl2ss_3dcv.get_calibration_rm(calibration_path, host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

uv2xy = calibration_lt.uv2xy
xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
depth_scale = 1000

xy1_o = xy1[:-1, :-1, :]
xy1_d = xy1[1:, 1:, :]

# Initialize PV intrinsics and extrinsics -------------------------------------
pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
pv_extrinsics = np.eye(4, 4, dtype=np.float32)

# Create readers --------------------------------------------------------------
rd_lt = hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, True)
sq_pv = hl2ss_io.sequencer(hl2ss_io.create_rd(os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)}.bin'), hl2ss.ChunkSize.SINGLE_TRANSFER, 'bgr24'))

rd_lt.open()
sq_pv.open()

# Main loop -------------------------------------------------------------------
cv2.namedWindow('RGBD')

index = 0

while (True):
    cv2.waitKey(1)

    # Read frames -------------------------------------------------------------
    data_lt = rd_lt.get_next_packet()
    if (data_lt is None):
        break
    if (not hl2ss.is_valid_pose(data_lt.pose)):
        print(f'Warning: invalid pose for {hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)} at {data_lt.timestamp}')
        continue

    status_pv, data_pv = sq_pv.get_next_packet(data_lt.timestamp)
    if (status_pv is None):
        break
    if (status_pv == hl2ss_mx.Status.DISCARDED):
        continue
    if (not hl2ss.is_valid_pose(data_pv.pose)):
        print(f'Warning: invalid pose for {hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)} at {data_pv.timestamp}')
        continue
    if (data_pv.payload.image is None):
        print(f'Warning: invalid image for {hl2ss.get_port_name(hl2ss.StreamPort.PERSONAL_VIDEO)} at {data_pv.timestamp}')
        continue

    # Preprocess frames -------------------------------------------------------
    depth = data_lt.payload.depth
    z     = hl2ss_3dcv.rm_depth_normalize(depth, scale)
    color = data_pv.payload.image

    pv_width  = color.shape[1]
    pv_height = color.shape[0]

    # Update PV intrinsics ----------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Generate depth map for PV image -----------------------------------------
    lt_to_world    = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
    world_to_pv    = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics)
    pv_to_pv_image = hl2ss_3dcv.camera_to_image(color_intrinsics)

    lt_points_o    = hl2ss_3dcv.rm_depth_to_points(xy1_o, z[:-1, :-1, :])        
    world_points_o = hl2ss_3dcv.transform(lt_points_o, lt_to_world)
    pv_points_o    = hl2ss_3dcv.transform(world_points_o, world_to_pv)
    pv_depth       = pv_points_o[:, :, 2:]
    pv_uv_o        = hl2ss_3dcv.project(pv_points_o, pv_to_pv_image)

    lt_points_d    = hl2ss_3dcv.rm_depth_to_points(xy1_d, z[:-1, :-1, :])
    world_points_d = hl2ss_3dcv.transform(lt_points_d, lt_to_world)
    pv_uv_d        = hl2ss_3dcv.project(world_points_d, world_to_pv @ pv_to_pv_image)

    pv_list_o     = hl2ss_3dcv.block_to_list(pv_uv_o)
    pv_list_d     = hl2ss_3dcv.block_to_list(pv_uv_d)
    pv_list_depth = hl2ss_3dcv.block_to_list(pv_depth)

    mask = (depth[:-1,:-1].reshape((-1,)) > 0)

    pv_list = np.hstack((np.floor(pv_list_o[mask, :]), np.floor(pv_list_d[mask, :]) + 1, pv_list_depth[mask]))
    pv_z    = np.zeros((pv_height, pv_width), dtype=np.float32)

    for n in range(0, pv_list.shape[0]):
        u0 = int(pv_list[n, 0])
        v0 = int(pv_list[n, 1])
        u1 = int(pv_list[n, 2])
        v1 = int(pv_list[n, 3])

        if ((u0 < 0) or (u0 >= pv_width)):
            continue
        if ((u1 < 0) or (u1 > pv_width)):
            continue
        if ((v0 < 0) or (v0 >= pv_height)):
            continue
        if ((v1 < 0) or (v1 > pv_height)):
            continue

        pv_z[v0:v1, u0:u1] = pv_list[n, 4]

    pv_z[pv_z < 0] = 0
    pv_z = (pv_z * depth_scale).astype(np.uint16)

    pv_z_colormap = hl2ss_3dcv.rm_depth_colormap(pv_z, 7500)

    # Write images ------------------------------------------------------------
    cv2.imwrite(os.path.join(path_out_pv, f'{index:06d}.png'), color)
    cv2.imwrite(os.path.join(path_out_lt, f'{index:06d}.png'), pv_z)
    cv2.imwrite(os.path.join(path_out_vz, f'{index:06d}.png'), pv_z_colormap)
    
    data_pv.pose.tofile(os.path.join(path_out_wp, f'{index:06d}.bin'))
    
    if (index == 0):
        color_intrinsics.tofile(os.path.join(path, 'pv_intrinsics.bin'))
        color_extrinsics.tofile(os.path.join(path, 'pv_extrinsics.bin'))

    index += 1

    # Display results ---------------------------------------------------------
    cv2.imshow('RGBD', np.hstack((color, pv_z_colormap)))

# Cleanup ---------------------------------------------------------------------
sq_pv.close()
rd_lt.close()

print(f'RGB saved to {path_out_pv}')
print(f'Depth saved to {path_out_lt}')
print(f'Depth (colormap) saved to {path_out_vz}')
print(f'Poses saved to {path_out_wp}')
print(f'Camera intrinsics and extrinsics saved to {path}')
