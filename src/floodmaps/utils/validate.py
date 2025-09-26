from dataclasses import dataclass
from typing import List, Tuple, Optional
from rasterio.crs import CRS
from rasterio.transform import Affine
from pathlib import Path
import rasterio
import logging

@dataclass
class ValidationResult:
    """Result of event file validation containing status and detailed information.
    
    Attributes
    ----------
    is_valid : bool
        Whether the event is valid.
    reference_shape : Tuple[int, int]
        The shape of the reference file.
    reference_crs : CRS
        The CRS of the reference file.
    reference_transform : Affine
        The affine transform of the reference file.
    mismatched_file : str
        Name of first file that does not match the reference shape, crs, or transform.
    error_message : Optional[str]
        Error message if the event is not valid.
    total_files : int
        Total number of .tif files validated in the event directory.
    """
    is_valid: bool
    reference_shape: Optional[Tuple[int, int]] = None
    reference_crs: Optional[CRS] = None
    reference_transform: Optional[Affine] = None
    mismatched_file: str = None
    error_message: Optional[str] = None
    total_files: int = 0

def validate_event_rasters(
        event_path: Path | str,
        logger: Optional[logging.Logger] = None
) -> ValidationResult:
    """
    Validate that all tiff files in an event directory have the same shape, CRS,
    and transform.
    
    Parameters
    ----------
    event_path : Path or str
        Path to the event directory containing tiff files.
    logger : logging.Logger, optional
        Logger for detailed information. If None, validation proceeds silently.
    
    Returns
    -------
    ValidationResult
        Object containing validation status and detailed information about any issues.
    """
    event_dir = Path(event_path)
    
    if not event_dir.exists():
        error_msg = f"Event directory does not exist: {event_path}"
        if logger:
            logger.error(error_msg)
        return ValidationResult(is_valid=False, error_message=error_msg)
    
    tif_files = list(event_dir.glob('*.tif'))
    
    if not tif_files:
        error_msg = f"No tiff files found in directory: {event_path}"
        if logger:
            logger.warning(error_msg)
        return ValidationResult(is_valid=False, error_message=error_msg, total_files=0)
    
    reference_shape = None
    reference_crs = None
    reference_transform = None
    
    if logger:
        logger.info(f"Validating shapes for {len(tif_files)} tiff files in {event_path}")
    
    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as src:
                current_shape = src.shape
                current_crs = src.crs
                current_transform = src.transform
                
                if reference_shape is None:
                    reference_shape = current_shape
                    reference_crs = current_crs
                    reference_transform = current_transform
                elif current_shape != reference_shape:
                    error_msg = f"Shape mismatch: {tif_path.name} has shape {current_shape}, expected {reference_shape}"
                    if logger:
                        logger.warning(error_msg)
                    return ValidationResult(is_valid=False,
                                            reference_shape=reference_shape,
                                            reference_crs=reference_crs,
                                            reference_transform=reference_transform,
                                            mismatched_file=str(tif_path.name),
                                            error_message=error_msg)   
                elif current_crs != reference_crs:
                    error_msg = f"CRS mismatch: {tif_path.name} has CRS {current_crs}, expected {reference_crs}"
                    if logger:
                        logger.warning(error_msg)
                    return ValidationResult(is_valid=False, 
                                            reference_shape=reference_shape,
                                            reference_crs=reference_crs,
                                            reference_transform=reference_transform,
                                            mismatched_file=str(tif_path.name),
                                            error_message=error_msg)
                elif not current_transform.almost_equals(reference_transform):
                    error_msg = f"Transform mismatch: {tif_path.name} has transform {current_transform}, expected {reference_transform}"
                    if logger:
                        logger.warning(error_msg)
                    return ValidationResult(is_valid=False,
                                            reference_shape=reference_shape,
                                            reference_crs=reference_crs,
                                            reference_transform=reference_transform,
                                            mismatched_file=str(tif_path.name),
                                            error_message=error_msg)
        except Exception as e:
            error_msg = f"Error reading {tif_path.name}: {str(e)}"
            if logger:
                logger.exception(error_msg)
            return ValidationResult(is_valid=False, error_message=error_msg, total_files=len(tif_files))
    
    if logger:
        logger.info(f"✓ All {len(tif_files)} files have consistent shape, crs, and transform.")
    
    return ValidationResult(
        is_valid=True,
        reference_shape=reference_shape,
        reference_crs=reference_crs,
        reference_transform=reference_transform,
        total_files=len(tif_files)
    )

# Convenience function for simple boolean check
def is_event_valid(event_path: str) -> bool:
    """Simple boolean check for event shape, crs, and transform validation."""
    return validate_event_rasters(event_path).is_valid