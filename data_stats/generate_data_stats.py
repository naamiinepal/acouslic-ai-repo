from functools import partial
from multiprocessing.pool import Pool
from typing import Dict

import tqdm
from acouslic_utils import *
from dotenv import dotenv_values
import yaml
from pathlib import Path
import numpy as np
import os

np.printoptions(suppress=True,precision=4)
# column names
header = [
    "filestem",
    "has_opt",
    "has_subopt",
    "opt_frame_count",
    "sweepwise_opt_frame_count",
    "subopt_frame_count",
    "sweepwise_subopt_frame_count",
    "opt_seq",
    "subopt_seq",
    "foreground_voxel_ratio",
    "opt_voxel_ratio",
    "subopt_voxel_ratio",
]


def process_acouslic_mask_path_helper(mask_path, log_dir, log_filename):

    stats_row = get_data_stats_formatted_row(get_stats_row(mask_path))

    with open(f"{log_dir}/{log_filename}", "a", encoding="utf-8") as f:
        f.write(f"{stats_row}\n")


def get_stats_row(mask_path):
    mask_np = sitk_to_numpy(read_image(mask_path)).astype(np.uint)
    global_foreground_voxel_ratio = np.sum(mask_np > 1) / np.prod(mask_np.shape)
    file_stem = get_stem(mask_path)

    subopt_seq = to_index_sequence_contiguous(
        get_suboptimal_frame_number(mask_np, consts["SUBOPTIMAL"])
    )
    opt_seq = to_index_sequence_contiguous(
        get_optimal_frame_number(mask_np, consts["OPTIMAL"])
    )

    has_opt = True if opt_seq else False
    has_subopt = True if subopt_seq else False
    opt_voxel_ratio = 0.0
    if opt_seq:
        opt_voxel_ratio = np.average([np.sum(mask_np[:,:,s:e+1]==int(consts['OPTIMAL'])) / np.prod(mask_np[:,:,s:e+1].shape) for s,e in opt_seq])
    subopt_voxel_ratio = 0.0
    if subopt_seq:
        subopt_voxel_ratio = np.average([np.sum(mask_np[:,:,s:e+1]==int(consts['SUBOPTIMAL']))  / np.prod(mask_np[:,:,s:e+1].shape) for s,e in subopt_seq])
    data_stats = {
        "filestem": file_stem,
        "has_opt": has_opt,
        "has_subopt": has_subopt,
        "opt_seq": opt_seq,
        "subopt_seq": subopt_seq,
        "foreground_voxel_ratio": global_foreground_voxel_ratio,
        "opt_voxel_ratio": opt_voxel_ratio,
        "subopt_voxel_ratio": subopt_voxel_ratio
    }

    # derived stats
    subopt_frame_count = [e - s + 1 for s, e in subopt_seq]
    opt_frame_count = [e - s + 1 for s, e in opt_seq]

    data_stats["opt_frame_count"] = sum(opt_frame_count)
    data_stats["subopt_frame_count"] = sum(subopt_frame_count)

    data_stats["sweepwise_opt_frame_count"] = get_seq_len_per_sweep(opt_seq)
    data_stats["sweepwise_subopt_frame_count"] = get_seq_len_per_sweep(subopt_seq)

    return data_stats


def write_csv_header(filepath, filename):
    """write output csv header"""
    outdir = Path(f"{filepath}/")
    outdir.mkdir(exist_ok=True)
    with open(outdir / f"{filename}", "w", encoding="utf-8") as f:
        header = get_data_stats_formatted_header()
        f.write(f"{header}\n")


def get_data_stats_formatted_header():
    header_str = ""
    for col_name in header:
        header_str += f"{col_name},"
    return header_str


def get_data_stats_formatted_row(data_stats: Dict):
    """output formatted string containing comma-separated landmarks"""
    row = ""
    for col_name in header:
        val = data_stats[col_name]
        if type(val) == list or type(val) == tuple or type(val) == dict:
            # format tuple so they are surrounded by double quote; disambiguate with csv delimiter
            out_str = f'"{val}",'  # surround with double quote
            row += out_str
        elif type(val) == float or type(val) == np.float64:
            row +=f"{val:.3f},"
        else:
            row += f"{val},"
    return row


def process_dir_multithreaded(path_list, args):

    print(f"processing {len(path_list)} files")

    write_csv_header(args.csv_dir, args.csv_filename)
    worker_fn = partial(
        process_acouslic_mask_path_helper,
        log_dir=args.csv_dir,
        log_filename=args.csv_filename,
    )
    num_workers = os.cpu_count() // 2  # use half of CPU core
    pool = Pool(processes=num_workers)
    jobs = []
    for item in path_list:
        job = pool.apply_async(worker_fn, (item,))
        jobs.append(job)
    for job in jobs:
        job.get()
    pool.close()
    pool.join()

    print("done")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir", type=str, required=True)
parser.add_argument("--csv_filename", type=str, required=True)
parser.add_argument("--test", action="store_true", default=False)

args = parser.parse_args()


env = dotenv_values(".env")
print(env)

consts = None
with open("consts.yaml") as f:
    try:
        consts = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


image_dir = Path(env["DS_RAW"]) / "images" / f'{consts["image_dir"]}'
image_paths = sorted(image_dir.rglob("*.mha"))

mask_dir = Path(env["DS_RAW"]) / "masks" / f'{consts["mask_dir"]}'
mask_paths = sorted(mask_dir.rglob("*.mha"))

print(f"reading image from {str(image_dir)}")
print(f"reading masks from {mask_dir}")

if args.test:
    to_process_mask_paths = mask_paths[:10]
else:
    to_process_mask_paths = mask_paths

process_dir_multithreaded(to_process_mask_paths, args)
