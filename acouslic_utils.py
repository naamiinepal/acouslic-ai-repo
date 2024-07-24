from pathlib import Path
import SimpleITK as sitk
import numpy as np
 
def read_image(img_path:Path):
    return sitk.ReadImage(str(img_path))

def write_image(img_path:Path):
    sitk.WriteImage()
def sitk_to_numpy(img:sitk.Image)->np.array:
    return sitk.GetArrayFromImage(img).swapaxes(0,-1) # swap axes to keep the size same

def numpy_to_sitk(img:np.array) -> sitk.Image:
    return sitk.GetImageFromArray(img.swapaxes(0,-1)) # swap axes to keep the size same

def get_stem(img_path: Path):
    return img_path.stem