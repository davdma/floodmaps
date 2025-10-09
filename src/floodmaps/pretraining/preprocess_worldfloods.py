import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Callable, List
from glob import glob
import rasterio
import concurrent.futures
import multiprocessing
from pathlib import Path
import json
import os
import logging
import sys
import traceback
from datetime import datetime

from floodmaps.pretraining.data_utils import WindowSize, WindowSlices, get_list_of_window_slices, filter_windows_v2

BANDS_S2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
BANDS_GT1 = ["Invalid", "Clear", "Cloud"]
BANDS_GT2 = ["Invalid", "Land", "Water"]

def process_filename_train_test(train_test_split_file, path_to_splits, input_folder: str="S2",
                                target_folder: str="gt"):
    if train_test_split_file:
        with open(train_test_split_file, "r") as fh:
            filenames_train_test = json.load(fh)
    else:
        assert (path_to_splits is not None) and os.path.exists(path_to_splits), \
            f"train_test_split_file not provided and path_to_splits folder {path_to_splits} does not exist"

        print(f"train_test_split_file not provided. We will use the content in the folder {path_to_splits}")
        filenames_train_test = {'train': {target_folder:[], input_folder:[]},
                                'test': {target_folder:[],input_folder:[]},
                                'val': {target_folder:[],input_folder:[]}}

    # loop through the naming splits
    for isplit in ["train", "test", "val"]:
        for foldername in [input_folder, target_folder]:

            # glob files in path_to_splits dir if there're not files in the given split
            if len(filenames_train_test[isplit][foldername]) == 0:
                # get the subdirectory
                assert (path_to_splits is not None) and os.path.exists(path_to_splits), \
                    f"path_to_splits {path_to_splits} doesn't exists or not provided and there're no files in split {isplit} folder {foldername}"

                path_2_glob = os.path.join(path_to_splits, isplit, foldername, "*.tif")
                filenames_train_test[isplit][foldername] = glob(path_2_glob)
                assert len(filenames_train_test[isplit][foldername]) > 0, f"No files found in {path_2_glob}"

        assert len(filenames_train_test[isplit][input_folder]) == len(filenames_train_test[isplit][target_folder]), \
            f"Different number of files in {input_folder} and {target_folder} for split {isplit}: {len(filenames_train_test[isplit][input_folder])} {len(filenames_train_test[isplit][target_folder])}"

        # check correspondence input output files (assert files exists)
        for idx, filename in enumerate(filenames_train_test[isplit][input_folder]):
            assert Path(filename).exists(), f"File input: {filename} does not exist"

            filename_target = filenames_train_test[isplit][target_folder][idx]
            assert Path(filename_target).exists(), f"File target: {filename_target} does not exist"

    return filenames_train_test

def preprocess(split:str,
                filenames_train_test:Dict[str, Any],
                window_size: Tuple[int, int] = (64, 64),
                channels=[3, 2, 1, 7],
                add_ndwi=True,
                workers:int = 10,
                filter_windows:Callable = None,
                threshold_missing:float = 0.0,
                image_prefix:str = "S2",
                gt_prefix:str = "gt",
                dir_path:str = None):
    """Deprecated window based preprocessing. Very slow so use batch preprocess instead.
    Note: Channels 3, 2, 1, 7 correspond to rgb+nir"""
    filenames = filenames_train_test[split][image_prefix]
    window_size = WindowSize(height=window_size[0], width=window_size[1])

    # first get the total number of window slices
    list_of_windows = get_list_of_window_slices(filenames, window_size=window_size)
    # filter window slices if needed
    if filter_windows is not None:
        list_of_windows = filter_windows(list_of_windows, threshold_missing = threshold_missing, image_prefix=image_prefix, gt_prefix=gt_prefix)

    logger = logging.getLogger('preprocessing')
    logger.debug(f"Number of windows: {len(list_of_windows)}")

    # allocate the total array (+1 for label)
    arr = np.zeros((len(list_of_windows), len(channels) + int(add_ndwi) + 1, window_size[0], window_size[1]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_window, arr, idx, window, channels, add_ndwi, image_prefix, gt_prefix) for idx, window in enumerate(list_of_windows)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # will re-raise any exception from the thread
            except Exception as e:
                print(f"Error in thread: {e}")
                raise e

    # save array to npy
    np.save(Path(dir_path) / f"{split}_patches.npy", arr)

def process_window(
                arr:np.ndarray,
                idx:int,
                window:WindowSlices,
                channels:List[int] = [3, 2, 1, 7],
                add_ndwi:bool = True,
                image_prefix:str = "S2",
                gt_prefix:str = "gt"
    ):
    """For parallel processing of windows. Deprecated so use batch preprocess instead."""
    # read in image
    # get filename
    image_name = window.file_name

    # replace string for image_prefix
    image_name = image_name.replace(gt_prefix, image_prefix, 1)

    # replace string for gt_prefix
    y_name = image_name.replace(image_prefix, gt_prefix, 1)

    # Get S2 image
    with rasterio.open(image_name) as src:
        all_channels_img = src.read(window=window.window, boundless=True, fill_value=0)
    
    selected_channels = all_channels_img[channels]
    selected_channels = selected_channels.astype(np.float32) / 10000

    # calculate ndwi
    if add_ndwi:
        green = all_channels_img[2].astype(np.float32)
        nir = all_channels_img[7].astype(np.float32)
        denominator = green + nir
        ndwi = np.full_like(green, -999999.0, dtype=np.float32)
        valid = (denominator != 0) & np.isfinite(green) & np.isfinite(nir)
        np.divide(green - nir, denominator, out=ndwi, where=valid)
        ndwi = np.expand_dims(ndwi, axis=0)
        selected_channels = np.vstack([selected_channels, ndwi])

    # Get labels
    with rasterio.open(y_name) as src:
        labels = src.read(window=window.window, boundless=True, fill_value=0)

    # reduce label to water or not water, get rid of invalid
    labels = (labels[1] == 2).astype(np.float32)
    labels = np.expand_dims(labels, axis=0)
    
    result = np.vstack([selected_channels, labels])
    arr[idx] = result

def _tile_nchw(img: np.ndarray, tile_size: int) -> np.ndarray:
    """Tile an image in NCHW format into (num_tiles, C, tile, tile) without Python loops.

    Discards remainder edges for speed.
    """
    C, H, W = img.shape
    h_tiles = H // tile_size
    w_tiles = W // tile_size
    if h_tiles == 0 or w_tiles == 0:
        return np.empty((0, C, tile_size, tile_size), dtype=img.dtype)
    Hc = h_tiles * tile_size
    Wc = w_tiles * tile_size
    img_cropped = img[:, :Hc, :Wc]
    # reshape and permute to tiles
    tiles = img_cropped.reshape(C, h_tiles, tile_size, w_tiles, tile_size)
    tiles = tiles.swapaxes(2, 3)  # C, h_tiles, w_tiles, tile, tile
    tiles = tiles.reshape(C, h_tiles * w_tiles, tile_size, tile_size)
    tiles = tiles.transpose(1, 0, 2, 3)  # (num_tiles, C, tile, tile)
    return tiles

def _process_files_worker(
        file_batch: List[str],
        channels: List[int],
        add_ndwi: bool,
        tile_size: int,
        image_prefix: str,
        gt_prefix: str,
        do_filter: bool,
        threshold_missing: float,
        threshold_clouds: float
    ) -> np.ndarray:
    """Worker that processes a batch of files: reads full rasters, tiles, filters, and concatenates.

    Returns a numpy array shaped (N_tiles, C_out, tile_size, tile_size).
    """
    tiles_accum: List[np.ndarray] = []
    for image_name in file_batch:
        try:
            # Read image and labels fully
            with rasterio.open(image_name) as src:
                img_all = src.read()  # (B, H, W)
            y_name = image_name.replace(image_prefix, gt_prefix, 1)
            with rasterio.open(y_name) as src:
                label_all = src.read()  # assume 2 bands

            # Select channels and scale
            img_selected = img_all[channels].astype(np.float32) / 10000.0  # (Cx, H, W)

            # NDWI over full image if requested (computed before tiling)
            if add_ndwi:
                green = img_all[2].astype(np.float32)
                nir = img_all[7].astype(np.float32)
                denom = green + nir
                ndwi = np.full_like(green, -999999.0, dtype=np.float32)
                valid = (denom != 0) & np.isfinite(green) & np.isfinite(nir)
                np.divide(green - nir, denom, out=ndwi, where=valid)
                x_full = np.concatenate([img_selected, ndwi[None, :, :]], axis=0)  # (Cx+1, H, W)
            else:
                x_full = img_selected  # (Cx, H, W)

            # Tile inputs once
            x_tiles = _tile_nchw(x_full, tile_size)  # (N, Cx(+1), t, t)

            # If there are no tiles, continue
            if x_tiles.shape[0] == 0:
                continue

            # Labels: compute invalid mask fraction using first two bands like previous filter
            # label_all expected shape (C, H, W)
            # Build filter mask if requested
            tile_mask = None
            # Tile labels once as a 2-channel stack
            lbl_full = label_all[:2]  # (2, H, W)
            lbl_tiles = _tile_nchw(lbl_full, tile_size)  # (N, 2, t, t)
            if do_filter:
                # invalid pixel where both bands == 0
                invalid_counts = ((lbl_tiles[:, 0] == 0) & (lbl_tiles[:, 1] == 0)).sum(axis=(1, 2)).astype(np.float32)
                frac_invalid = invalid_counts / (tile_size * tile_size)

                # cloud pixels are where band 0 == 2
                cloud_counts = (lbl_tiles[:, 0] == 2).sum(axis=(1, 2)).astype(np.float32)
                frac_cloud = cloud_counts / (tile_size * tile_size)

                tile_mask = (frac_invalid <= threshold_missing) & (frac_cloud <= threshold_clouds)
            else:
                tile_mask = np.ones((x_tiles.shape[0],), dtype=bool)

            # Water label from second band == 2, as in previous implementation
            water_tiles_bin = (lbl_tiles[:, 1] == 2).astype(np.float32)  # (N, t, t)
            water_tiles_bin = water_tiles_bin[:, None, :, :]  # (N, 1, t, t)

            # Assemble output channels: selected + optional ndwi + label
            out_tiles = np.concatenate([x_tiles, water_tiles_bin], axis=1)

            # Apply mask
            out_tiles = out_tiles[tile_mask]

            if out_tiles.shape[0] > 0:
                tiles_accum.append(out_tiles)
        except Exception as e:
            # Capture and return detailed traceback through the raised exception
            tb = traceback.format_exc()
            print(f"Worker error in file {image_name}: {e}\n{tb}", file=sys.stderr, flush=True)
            raise

    if len(tiles_accum) == 0:
        C_out = len(channels) + (1 if add_ndwi else 0) + 1
        return np.empty((0, C_out, tile_size, tile_size), dtype=np.float32)

    return np.concatenate(tiles_accum, axis=0)

def preprocess_batch(
                split: str,
                filenames_train_test: Dict[str, Any],
                window_size: Tuple[int, int] = (64, 64),
                channels: List[int] = [3, 2, 1, 7],
                add_ndwi: bool = True,
                workers: int = 10,
                filter_windows: Callable = None,
                threshold_missing: float = 0.0,
                threshold_clouds: float = 0.5,
                image_prefix: str = "S2",
                gt_prefix: str = "gt",
                dir_path: str = None
    ) -> None:
    """Efficient preprocessing: assign files to workers, tile in memory, filter, and concatenate.

    Saves a single NPY file per split with shape (N, C, tile, tile).
    """
    logger = logging.getLogger('preprocessing')
    filenames = filenames_train_test[split][image_prefix]
    total_files = len(filenames)
    tile_size = window_size[0]

    if total_files == 0:
        logger.warning(f"No files found for split {split}")
        np.save(Path(dir_path) / f"{split}_patches.npy", np.empty((0, len(channels) + int(add_ndwi) + 1, tile_size, tile_size), dtype=np.float32))
        return

    # Split files into roughly equal chunks per worker
    num_workers = max(1, workers)
    # Do not create more chunks than files
    num_chunks = min(num_workers, total_files)
    chunks = np.array_split(np.array(filenames, dtype=object), num_chunks)
    chunks = [list(c) for c in chunks if len(c) > 0]

    logger.info(f"Batch preprocessing {split}: {total_files} files across {len(chunks)} workers")

    # Progress printing every 1%
    next_pct = 1
    processed_files = 0

    results: List[np.ndarray] = []

    def _update_progress(delta: int):
        nonlocal processed_files, next_pct
        processed_files += delta
        pct = int((processed_files * 100) / total_files)
        while pct >= next_pct and next_pct <= 100:
            logger.info(f"{split}: {processed_files}/{total_files} files ({next_pct}%) processed")
            next_pct += 1

    # Use 'spawn' start method to avoid issues with forking GDAL/rasterio, and recycle workers
    mp_ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp_ctx, max_tasks_per_child=1) as executor:
        futures = [
            executor.submit(
                _process_files_worker,
                chunk,
                channels,
                add_ndwi,
                tile_size,
                image_prefix,
                gt_prefix,
                filter_windows is not None,
                threshold_missing,
                threshold_clouds,
            )
            for chunk in chunks
        ]

        for fut, chunk in zip(concurrent.futures.as_completed(futures), chunks):
            try:
                arr = fut.result()
                results.append(arr)
            finally:
                _update_progress(len(chunk))

    # Final concatenate and save
    if len(results) == 0:
        raise Exception("Result is empty.")
    else:
        out = np.concatenate(results, axis=0)

    logger.info(f"{split}: produced {out.shape[0]} tiles with tensor shape {out.shape}")
    np.save(Path(dir_path) / f"{split}_patches.npy", out)

def run_preprocess(cfg: DictConfig) -> None:
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    DATASET_PATH = cfg.pretraining.dataset_path
    CSV_PATH = os.path.join(DATASET_PATH,"dataset_metadata.csv")
    JSON_PATH = os.path.join(DATASET_PATH, "train_test_split_from_csv.json")

    # Create timestamp for logging
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'''Starting World Floods S2 pretrain dataset preprocessing:
        Date:            {timestamp}
        Dataset path:    {DATASET_PATH}
        CSV path:        {CSV_PATH}
        JSON path:       {JSON_PATH}
        Patch size:      {cfg.pretraining.window_size}
        Workers:         {cfg.pretraining.workers}
        Filter windows:  {cfg.pretraining.filter_windows}
        Threshold missing: {getattr(cfg.pretraining, 'threshold_missing', 0.0)}
        Image prefix:    {cfg.pretraining.image_prefix}
        GT prefix:       {cfg.pretraining.gt_prefix}
        Dir path:        {cfg.pretraining.dir_path}
    ''')

    def convert_metadata_csv_to_json() -> None:
        out: Dict[str, Any] = {}
        modalities = ["S2", "gt"]
        csv = pd.read_csv(CSV_PATH)

        for split in csv.split.unique():
            out[split] = {}
            files = csv[csv.split == split]["event id"]
            for mod in modalities:
                out[split][mod] = [
                    os.path.join(DATASET_PATH, split, mod, f"{fn}.tif")
                    for fn in files.to_list()
                ]

        with open(JSON_PATH, "w") as f:
            json.dump(out, f, indent=2)

    convert_metadata_csv_to_json()

    path_to_splits = DATASET_PATH
    train_test_split_file = JSON_PATH
    filenames_train_test = process_filename_train_test(train_test_split_file, path_to_splits)

    # make the preprocess dir path if it doesn't exist
    Path(cfg.pretraining.dir_path).mkdir(parents=True, exist_ok=True)

    # save the cfg.pretraining dict to json in the preprocess dir path
    container = OmegaConf.to_container(cfg.pretraining, resolve=True, enum_to_str=True)
    with open(Path(cfg.pretraining.dir_path) / "config.json", "w") as f:
        json.dump(container, f, indent=2)

    filter_windows = filter_windows_v2 if cfg.pretraining.filter_windows else None

    logger.info("Preprocessing train set (batch mode)")
    preprocess_batch("train",
                filenames_train_test,
                window_size=cfg.pretraining.window_size,
                channels=cfg.pretraining.channels,
                add_ndwi=cfg.pretraining.add_ndwi,
                workers=cfg.pretraining.workers,
                filter_windows=filter_windows,
                threshold_missing=getattr(cfg.pretraining, 'threshold_missing', 0.0),
                threshold_clouds=getattr(cfg.pretraining, 'threshold_clouds', 0.5),
                image_prefix=cfg.pretraining.image_prefix,
                gt_prefix=cfg.pretraining.gt_prefix,
                dir_path=cfg.pretraining.dir_path)
    
    logger.info("Preprocessing val set (batch mode)")
    preprocess_batch("val",
                filenames_train_test,
                window_size=cfg.pretraining.window_size,
                channels=cfg.pretraining.channels,
                add_ndwi=cfg.pretraining.add_ndwi,
                workers=cfg.pretraining.workers,
                filter_windows=filter_windows,
                threshold_missing=getattr(cfg.pretraining, 'threshold_missing', 0.0),
                threshold_clouds=getattr(cfg.pretraining, 'threshold_clouds', 0.5),
                image_prefix=cfg.pretraining.image_prefix,
                gt_prefix=cfg.pretraining.gt_prefix,
                dir_path=cfg.pretraining.dir_path)
    
    logger.info("Preprocessing test set (batch mode)")
    preprocess_batch("test",
                filenames_train_test,
                window_size=cfg.pretraining.window_size,
                channels=cfg.pretraining.channels,
                add_ndwi=cfg.pretraining.add_ndwi,
                workers=cfg.pretraining.workers,
                filter_windows=filter_windows,
                threshold_missing=getattr(cfg.pretraining, 'threshold_missing', 0.0),
                threshold_clouds=getattr(cfg.pretraining, 'threshold_clouds', 0.5),
                image_prefix=cfg.pretraining.image_prefix,
                gt_prefix=cfg.pretraining.gt_prefix,
                dir_path=cfg.pretraining.dir_path)


@hydra.main(version_base=None, config_path='pkg://configs', config_name='config.yaml')
def main(cfg: DictConfig) -> None:
    run_preprocess(cfg)

if __name__ == "__main__":
    main()