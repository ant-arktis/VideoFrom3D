import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from os.path import join
from os import makedirs
from pathlib import Path
import argparse
from controlnet_aux import HEDdetector


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='sample001'
                   # required=True,
                   )
    return p.parse_args()


def main():
    args = parse()
    model = HEDdetector.from_pretrained("lllyasviel/Annotators").to('cuda')
    for path_arc in sorted(glob(join(args.target, "arc*/normal"))):
        path_hed = Path(path_arc).parent / 'hed'
        makedirs(path_hed, exist_ok=True)
        target_files = sorted(glob(join(path_arc, "*.png")))
        for target_file in tqdm(target_files):
            im = Image.open(target_file)
            im = im.convert('RGB')
            size = im.size
            im = model(im, detect_resolution=size[1], image_resolution=size[1])
            im = im.resize(size)
            path_output = join(path_hed, Path(target_file).name)
            im.save(path_output, quality=100, subsampling=0)


if __name__ == "__main__":
    main()
