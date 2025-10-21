# Satellite Data Preprocessing Scripts

This directory contains scripts for preprocessing satellite imagery and labels into smaller sized training patches for flood detection models. The preprocessing pipeline handles both manual and machine-generated labels with parallel processing capabilities for large-scale datasets.

### Script Types

1. **Manual Label Preprocessing**: `preprocess_s2.py`
   - Processes manually labeled tiles (human-annotated ground truth)
   - Uses hardcoded train/val/test label splits (YAML defined)
   - Sequential processing suitable for smaller datasets
   - Applies `BOA_ADD_OFFSET` correction to S2 bands from 2022-01-25 onwards

2. **Weak Label Preprocessing**: `preprocess_s2_weak.py` and `preprocess_sar_weak.py`
   - Processes machine-generated prediction tiles with optional manual label substitution
   - **Requires `pred_*.tif` files**: Generated using [`../inference/weak_labels.py`](../inference/weak_labels.py) script
   - Find valid tiles in list of sample directories (YAML defined)
   - Handles large-scale datasets (10,000+ tiles) using concurrent workers
   - S2 preprocessing applies `BOA_ADD_OFFSET` correction to bands from 2022-01-25 onwards

### Prerequisites for Weak Label Preprocessing

Before running weak label preprocessing scripts, you must generate machine labels using the `weak_labels.py` script. This creates `pred_YYYYMMDD_YYYYMMDD_Y_X.tif` files in each event directory that the preprocessing scripts will use as labels.

## Data Structure

### SAR Preprocessing (`preprocess_sar_weak.py`)
**Output**: 13-channel patches
- Channels 1-2: SAR VV/VH polarization
- Channels 3-5: DEM, slope_y, slope_x
- Channels 6-8: waterbody, roads, flowlines (binary)
- Channel 9: binary flood label
- Channels 10-12: TCI (RGB scaled to [0,1])
- Channel 13: NLCD land cover

### S2 Preprocessing (`preprocess_s2_weak.py`)
**Output**: 22-channel patches
- Channels 1-3: RGB (reflectance scaled by 10000)
- Channel 4: B08 NIR (reflectance scaled by 10000)
- Channel 5: SWIR1/B11 (reflectance scaled by 10000)
- Channel 6: SWIR2/B12 (reflectance scaled by 10000)
- Channel 7: NDWI (unnormalized, range [-1, 1])
- Channel 8: MNDWI (unnormalized, range [-1, 1])
- Channel 9: AWEI_sh (unnormalized)
- Channel 10: AWEI_nsh (unnormalized)
- Channels 11-13: DEM, slope_y, slope_x
- Channels 14-16: waterbody, roads, flowlines (binary)
- Channel 17: binary flood label
- Channels 18-20: TCI (RGB scaled to [0,1])
- Channel 21: NLCD land cover
- Channel 22: SCL (Scene Classification Layer)

**Processing Baseline Offset Correction**: 
- ⚠️ **Critical**: S2 L2A products from 2022-01-25 onwards require `BOA_ADD_OFFSET` correction (-1000) to convert from scaled integer values to surface reflectance
- The preprocessing scripts automatically apply this offset based on tile date
- **Provider Consistency Warning**: Ensure the STAC provider does not already apply the offset. AWS Element 84 has known issues where products falsely indicate offset is not applied when it already is, leading to double-correction. Use Microsoft Planetary Computer (MPC) or Copernicus Data Space Ecosystem (CDSE) for reliable offset handling

### Configuration Files

#### Manual Label Configuration (`preprocess_manual.yaml`)
```yaml
s2:
  sample_dirs:
    - "s2_200_5_4_35/"
    - "s2_additional_samples/"
  
  label_splits:
    train:
      - "labels/label_20200318_20200318_12_34.tif"
      - "labels/label_20200415_20200415_15_28.tif"
      # ... more training labels
    val:
      - "labels/label_20200520_20200520_18_42.tif"
      # ... validation labels  
    test:
      - "labels/label_20200622_20200622_21_15.tif"
      # ... test labels
```

#### Weak Label Configuration (`preprocess.yaml`)
```yaml
s1:  # SAR configuration
  sample_dirs:
    - "s2_s1_200_6_4_10/"
    - "sar_additional_samples/"
  
  label_dirs:
    - "labels_texas/"
    - "labels_illinois/"
  
  split:
    seed: 433002
    val_ratio: 0.1
    test_ratio: 0.1

s2:  # Sentinel-2 configuration  
  sample_dirs:
    - "s2_200_5_4_35/"
    - "s2_additional_samples/"
  
  label_dirs:
    - "labels_texas/"
    - "labels_illinois/"
  
  split:
    seed: 433002
    val_ratio: 0.1
    test_ratio: 0.1
```

## Output Structure

```
data/preprocess/
├── s1_weak/
│   └── samples_64_1000_raw/
│       ├── train_patches.npy
│       ├── val_patches.npy 
│       ├── test_patches.npy
│       └── mean_std_64_1000_raw.pkl
├── s2_weak/
│   └── samples_64_1000/
│       ├── train_patches.npy
│       ├── val_patches.npy 
│       ├── test_patches.npy
│       └── mean_std_64_1000.pkl
└── s2/
    └── samples_64_1000/
        ├── train_patches.npy
        ├── val_patches.npy
        ├── test_patches.npy
        └── mean_std_64_1000.pkl
```

## Developer Notes

There are some important implementation details with regards to available RAM memory on the node (or computer). The datasets can get quite large, if you are sampling `1000` 64 by 64 patches with `2000` tiles with `16` channels per tile you can end up with a `1000 * 2000 * 16 * 64 * 64 * 4 / 1024**3 = 488 GB` sized numpy array. On LCRC improv nodes, the standard nodes have memory of `~233GB` regardless of `ncpus` requested, so this will not fit all on memory. The solution is to have two strategies depending on whether the arrays will fit in memory. If the array fits, each worker creates their own array in memory and the results are concatenated. If the array does not fit, each worker writes to a memory mapped array, which is then combined by writing them individually into a much larger memory mapped array.

⚠️ **Important**: While preprocessing supports generating much larger than memory `.npy` output files, the current `Dataset` classes assume they still fit in memory during training. If your final dataset exceeds available RAM, you'll need to implement memory-mapped data loading in your training pipeline (just add a `mmap_mode='r'` argument, but currently not implemented).

Some tips:
1. **Worker Count**: Use `n_workers` equal to available CPU cores for optimal performance.
2. **Memory Monitoring**: Monitor memory usage during processing; script will automatically choose appropriate strategy, but if strained consider using a smaller `num_samples` argument.
3. **Scratch Space**: The temp mem mapped arrays are saved on `/scratch/floodmapspre` directory. Ensure sufficient scratch disk space for large datasets.