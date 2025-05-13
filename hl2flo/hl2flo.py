
import numpy as np
import struct
import os


def scan_files(path, sort=True):
    items = os.listdir(path)
    paths = [os.path.join(path, item) for item in items]
    files = [path for path in paths if (os.path.isfile(path))]
    return sorted(files) if (sort) else files


def get_file_name(path):
    root, tail = os.path.split(path)
    name, ext = os.path.splitext(tail)
    return (root, name, ext)


def flow_to_flo(flow_uv, filename):
    with open(filename, 'wb') as f:
        f.write('PIEH'.encode('ascii'))
        f.write(struct.pack('<II', flow_uv.shape[1], flow_uv.shape[0]))
        f.write(flow_uv.astype(np.float32).tobytes())


def flo_to_flow(filename):
    with open(filename, 'rb') as f:
        f.read(4)
        width, height = struct.unpack('<II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.float32).reshape((height, width, 2))
    
