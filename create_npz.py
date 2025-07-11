# create a .npz file from a folder of image samples.
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--images_path", type=str, required=True)
parser.add_argument("--images_ext", type=str, default="jpg")
parser.add_argument("--result_file", type=str, default="images.npz")
parser.add_argument("--range_begin", type=int, default=0)
parser.add_argument("--range_end", type=int, default=1000)
args = parser.parse_args()
    

sample_dir = '../../../data/celeba_hq/data128x128/train/unlabeled'
samples = []
for i in tqdm(range(args.range_begin, args.range_end), desc="Building .npz file from samples"):
    sample_pil = Image.open(f"{args.images_path}/{i:05d}.{args.images_ext}").resize((64, 64))
    sample_np = (np.asarray(sample_pil)/255.).astype(np.float32)  # .astype(np.uint8)
    samples.append(sample_np)

samples = np.stack(samples)
np.savez(args.result_file, arr_0=samples)
print(f"Saved .npz file to {args.result_file} [shape={samples.shape}].")

