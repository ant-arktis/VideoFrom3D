import numpy as np
import cv2 as cv
from PIL import Image

if __name__ == "__main__":
    bit = 16
    lb = 0
    ub = 2**bit - 1
    res = 1024
    idxs = np.round(np.linspace(lb, ub, res)).astype(f'uint{bit}')
    xyz = np.meshgrid(idxs, idxs, [ub])
    rgb = np.concatenate(xyz, axis=-1).astype(f'uint{bit}')[::-1, :, ::-1]
    cv.imwrite(f'uv_{bit}_{res}.tif', rgb)
