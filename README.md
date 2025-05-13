# hl2flo

Tool for creating optical flow datasets using HoloLens 2.

This tool depends on the HoloLens positional tracking to perform depth alignment and optical flow estimation.
For best results, please follow the practices described in [HoloLens environment considerations](https://learn.microsoft.com/en-us/hololens/hololens-environment-considerations).

## Usage

1. Install [hl2ss](https://github.com/jdibenes/hl2ss) on your HoloLens.
2. Run [dataset_step_01_capture_video.py](hl2flo/dataset_step_01_capture_video.py) to capture a video sequence comprised of RGB data from the HoloLens front camera and unaligned depth data from the HoloLens ToF sensor.
3. Run [dataset_step_02_generate_rgbd.py](hl2flo/dataset_step_02_generate_rgbd.py) to extract RGB frames, intrinsics, extrinsics, poses, and generate aligned depth frames.
4. Run [dataset_step_03_generate_flow.py](hl2flo/dataset_step_03_generate_flow.py) to generate optical flows.
5. (Optional) Run [dataset_step_04_generate_disparity.py](hl2flo/dataset_step_04_generate_disparity.py) to convert aligned depth frames to disparity (horizontal flow).

Repeat steps 2 through 5 for each sequence.
