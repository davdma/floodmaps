import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class DataConfig:
    """Configuration class for managing data file paths and settings for
    sampling script."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from YAML file or use defaults.
        
        Parameters
        ----------
        config_file : str, optional
            Path to YAML configuration file. If None, looks for default config.
        """
        self.config_file = config_file or "configs/sample_s2_s1.yaml"
        self.config = self.load_config()
        self.paths = self.config.get('paths', {})
        self.validate_paths()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_paths(self):
        """Validate that all required data paths exist."""
        required_paths = [
            self.paths.get('prism_data'),
            self.paths.get('ceser_boundary'),
            self.paths.get('prism_meshgrid'),
            self.paths.get('nhd_wbd'),
            self.paths.get('elevation_dir'),
            self.paths.get('nlcd_dir')
        ]
        
        missing = []
        for path in required_paths:
            if path is None:
                raise KeyError(f"Required path not found in configuration.")
            if path and not Path(path).exists():
                missing.append(path)
        
        if missing:
            raise FileNotFoundError(
                f"Missing required data files/directories:\n" + "\n".join(missing)
            )
    
    def get_path(self, key: str, required: bool = True) -> str:
        """
        Get a specific path from configuration.
        
        Parameters
        ----------
        key : str
            Path key from config file
        required : bool
            Whether the path is required to exist
            
        Returns
        -------
        str
            Path value
        """
        path = self.paths.get(key)
        
        if path is None:
            if required:
                raise KeyError(f"Required path '{key}' not found in configuration")
            return ""
        
        if required and not Path(path).exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
        
        return path
    
    @property
    def prism_file(self) -> str:
        """Get PRISM data file path."""
        return self.get_path('prism_data')
    
    @property
    def ceser_boundary_file(self) -> str:
        """Get CESER boundary file path."""
        return self.get_path('ceser_boundary')
    
    @property
    def prism_meshgrid_file(self) -> str:
        """Get PRISM meshgrid file path."""
        return self.get_path('prism_meshgrid')
    
    @property
    def nhd_wbd_file(self) -> str:
        """Get NHD WBD file path."""
        return self.get_path('nhd_wbd')
    
    @property
    def elevation_directory(self) -> str:
        """Get elevation directory path."""
        return self.get_path('elevation_dir')
    
    @property
    def nlcd_directory(self) -> str:
        """Get NLCD directory path."""
        return self.get_path('nlcd_dir')
    
    @property
    def roads_directory(self) -> str:
        """Get roads directory path."""
        return self.get_path('roads_dir', required=False) 