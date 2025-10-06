from typing import List, Callable
from tqdm import tqdm
import numpy as np
import concurrent.futures
from itertools import product
import rasterio
from rasterio import windows
import json
from collections import namedtuple
from typing import Dict

WindowSize = namedtuple("WindowSize", ["height", "width"])
WindowSlices = namedtuple("WindowSlices", ["file_name", "window"])

def load_windows(filename:str) -> List[WindowSlices]:
    with open(filename, "r") as fh:
        list_of_windows = [Dict_to_WindowSlices(dictio) for dictio in json.load(fh)["slices"]]
    return list_of_windows

def WindowSlices_to_Dict(ws: WindowSlices) -> Dict:
    return {
        "file_name" : ws.file_name,
        "window": {
            "col_off" : ws.window.col_off,
            "row_off": ws.window.row_off,
            "width": ws.window.width,
            "height": ws.window.height,
        }
    }

def Dict_to_WindowSlices(ds: Dict) -> WindowSlices:
    return WindowSlices(file_name=ds["file_name"],
                        window=windows.Window(col_off=ds["window"]["col_off"],
                                              row_off=ds["window"]["row_off"],
                                              width=ds["window"]["width"],
                                              height=ds["window"]["height"]))

def get_window_tiles(
    ds: rasterio.io.DatasetReader, height: int = 64, width: int = 64, **kwargs
) -> List[rasterio.windows.Window]:
    """a generator for rasterio specific slices given a rasterio dataset

    Args:
        ds (rasterio.io.DatasetReader): a rasterio dataset object
        height (int): the height for the slice
        width (int): the width for the slice

    Yields:
        window (rasterio.windows.Window): slicing
    """
    # extract the row height from the dataset
    n_columns, n_rows = ds.meta["width"], ds.meta["height"]

    # create the offsets
    offsets = product(range(0, n_columns, width), range(0, n_rows, height)) # discrete tiling!
    list_of_windows = []
    for col_offset, row_offset in offsets:
        iwindow = windows.Window(
            col_off=col_offset, row_off=row_offset, width=width, height=height, **kwargs
        )
        list_of_windows.append(iwindow)

    return list_of_windows

def get_list_of_window_slices(
    file_names: List[str], window_size: WindowSize
) -> List[WindowSlices]:
    """Function to return the list of window slices for the all the
    input images and the given window size.

    Args:
        file_names (List[str]): List of filenames that are to be sliced.
        window_size (WindowSize): Window size of the tiles.

    Returns:
        List[WindowSlices]: List of window slices for the each tile
        corresponding to each input image.
    """

    accumulated_list_of_windows = []
    for ifilename in file_names:

        with rasterio.open(ifilename) as dataset:
            # get list of windows
            list_of_windows = get_window_tiles(
                dataset, height=window_size.height, width=window_size.width
            )
            # create a list of filenames
            list_of_windows = [
                WindowSlices(file_name=ifilename, window=iwindow)
                for iwindow in list_of_windows
            ]

        accumulated_list_of_windows += list_of_windows

    return accumulated_list_of_windows

def _filter_windows(
    fun_frac_invalids: Callable,
    list_of_windows: List[WindowSlices],
    threshold_missing: float = 0.5,
    image_prefix: str = "S2",
    gt_prefix: str = "gt",
    num_workers: int = 8,
) -> List[WindowSlices]:
    """Filter windows from the dataset with more than threshold_missing * 100 of missing pixels, in parallel."""

    def check_valid(sub_window: WindowSlices) -> bool:
        y_name = sub_window.file_name.replace(image_prefix, gt_prefix, 1)
        try:
            with rasterio.open(y_name) as src:
                label = src.read(window=sub_window.window, boundless=True, fill_value=0)
            frac_invalids = fun_frac_invalids(label)
            return frac_invalids <= threshold_missing
        except Exception as e:
            # Optionally log the error here
            return False

    valid_slices = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(check_valid, list_of_windows),
                total=len(list_of_windows),
                desc="Filtering invalid and cloudy windows (parallel)",
            )
        )
    valid_slices = [win for win, is_valid in zip(list_of_windows, results) if is_valid]
    return valid_slices

def filter_windows_v2(
    list_of_windows: List[WindowSlices],
    threshold_missing: float = 0.5,
    image_prefix: str = "S2",
    gt_prefix: str = "gt",
    num_workers: int = 8,
) -> List[WindowSlices]:
    """Filter windows from the dataset with more than threshold_missing * 100 of missing pixels, in parallel."""

    # Assumes first channel is cloud, second channel is water
    return _filter_windows(
        lambda label: ((label[0] == 0) & (label[1] == 0)).sum() / np.prod(label.shape[1:]),
        list_of_windows,
        threshold_missing=threshold_missing,
        image_prefix=image_prefix,
        gt_prefix=gt_prefix,
        num_workers=num_workers,
    )