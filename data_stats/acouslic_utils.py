from pathlib import Path
import SimpleITK as sitk
import numpy as np
from typing import List


# file i/o
def read_image(img_path: Path):
    return sitk.ReadImage(str(img_path))


def write_image(img: sitk.Image, img_path: Path):
    sitk.WriteImage(img, str(img_path), compressionLevel=9, useCompression=True)


def sitk_to_numpy(img: sitk.Image) -> np.array:
    return sitk.GetArrayFromImage(img).swapaxes(
        0, -1
    )  # swap axes to keep the size same


def numpy_to_sitk(img: np.array) -> sitk.Image:
    return sitk.GetImageFromArray(
        img.swapaxes(0, -1)
    )  # swap axes to keep the size same


def get_stem(img_path: Path):
    return img_path.stem


# sweeps, frame-index
def frame_index_to_sweep_id(frame_index, total_frames, TOTAL_SWEEPS=6):
    frames_per_sweeps = total_frames // TOTAL_SWEEPS
    return {
        "sweep_id": frame_index // frames_per_sweeps + 1,  # sweep_id starts from 1
        "frame_id": frame_index % frames_per_sweeps,
    }


def get_frame_number_with_foreground_val(img: np.array, foreground_val: int):
    """assume a 3D array with the last channel representing the frame count

    return the frame index range (frame with foreground segmentation value of `foreground_val`)
    """
    return np.where(img.max(axis=(0, 1)) == foreground_val)[0]


def get_optimal_frame_number(img: np.array, optimal_label: int):
    """assume a 3D array with the last channel representing the frame count

    return the optimal frame index range (frame with foreground segmentation of 1)"""
    return get_frame_number_with_foreground_val(img, foreground_val=optimal_label)


def get_suboptimal_frame_number(img: np.array, suboptimal_label: int):
    """assume a 3D array with the last channel representing the frame count

    return the optimal frame index range (frame with foreground segmentation of 1)"""
    return get_frame_number_with_foreground_val(img, foreground_val=suboptimal_label)


def to_index_sequence_contiguous(frame_indices: np.array, verbose=False):
    """
    input: [60,61,62,63, 612,613,614]
    output: [(60,63),(612-614)]
    """
    frame_start_indices = []
    frame_end_indices = []

    i = 0
    while i < len(frame_indices) - 1:
        current_frame = frame_indices[i]
        start_frame = frame_indices[i]
        while (i + 1) < len(frame_indices) and frame_indices[i + 1] == (
            current_frame + 1
        ):  # next frame index is current_frame + 1
            current_frame += 1  # increment current frame
            i += 1  # increment i until the sequence is contigous

        end_frame = frame_indices[i]  # i points to the end of contigous sequence

        frame_start_indices.append(start_frame)
        frame_end_indices.append(end_frame)

        i += 1  # skip to next contiguous sequence

        if verbose:
            print(f" start {start_frame} end {end_frame}")

    return zip_lists(frame_start_indices, frame_end_indices)


def zip_lists(a, b):
    return [(s, e) for s, e in zip(a, b)]  # (1,2,3,4) (a,b,c,d) -> [(1,a), (2,b)...]


def get_seq_len_per_sweep(seq:List):
    seq_lens = {}
    for s,e in seq:
        seq_len = e-s+1
        start_idx_dict = frame_index_to_sweep_id(s,840,6)
        end_idx_dict = frame_index_to_sweep_id(e,840,6)
        seq_lens[start_idx_dict['sweep_id']] = seq_len
    return seq_lens