#------------------------------------------------------------------------------
# Step 04 (Optional): Convert depth to disparity.
#------------------------------------------------------------------------------

import numpy as np
import os
import cv2
import flow_viz
import hl2ss_imshow
import hl2ss
import hl2ss_3dcv
import hl2flo

# Settings --------------------------------------------------------------------

# Data folder
path = './data/HL2-2025-05-13-02-38-33/'

# Baseline (in meters)
baseline = 0.2

# Minimum valid depth
min_depth = 0.1

# Disparity sign (-1: right, 1: left)
dx_sign = -1

#------------------------------------------------------------------------------

# Data Load -------------------------------------------------------------------
lt_in            = os.path.join(path, f'{hl2ss.get_port_name(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)}')
files_lt         = hl2flo.scan_files(lt_in)
color_intrinsics = np.fromfile(os.path.join(path, 'pv_intrinsics.bin'), dtype=np.float32).reshape((4, 4))

# Create output folder --------------------------------------------------------
lt_out     = os.path.join(path, f'disp')
lt_out_vis = os.path.join(path, f'disp_vis')

os.makedirs(lt_out)
os.makedirs(lt_out_vis)

# Main loop -------------------------------------------------------------------
max_lt = -1
max_dx = -1

index = 0

for file_lt in files_lt:
    # Convert depth to disparity ----------------------------------------------
    depth = cv2.imread(file_lt, cv2.IMREAD_UNCHANGED)

    lt = (depth.astype(np.float32)) / 1000

    local_max_lt = np.max(lt)
    if (local_max_lt > max_lt):
        max_lt = local_max_lt
    
    lt[lt < min_depth] = np.Inf

    dx = (baseline * color_intrinsics[0, 0]) / lt

    local_max_dx = np.max(dx)
    if (local_max_dx > max_dx):
        max_dx = local_max_dx

    flo = np.dstack((dx_sign * dx, np.zeros(dx.shape)))

    flo_vis = flow_viz.flow_to_image(flo, convert_to_bgr=True)

    # Write data --------------------------------------------------------------
    hl2flo.flow_to_flo(flo, os.path.join(lt_out, f'{index:06d}.flo'))
    cv2.imwrite(os.path.join(lt_out_vis, f'{index:06d}.png'), flo_vis)

    index += 1

    # Display results ---------------------------------------------------------
    cv2.imshow('Depth', np.hstack((hl2ss_3dcv.rm_depth_colormap(depth, 7500), flo_vis)))
    cv2.waitKey(1)

# End -------------------------------------------------------------------------
print(f'Max depth: {max_lt} | Max disparity: {max_dx}')

print(f'Disparity saved to {lt_out}')
print(f'Disparity (color) saved to {lt_out_vis}')
