"""
STAC Catalog Provider Abstraction

This module provides abstract base classes and concrete implementations for different
STAC catalog providers, allowing easy switching between Microsoft Planetary Computer,
AWS, and other providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import pystac_client
import planetary_computer
from pystac import Item, ItemCollection
from pystac.extensions.projection import ProjectionExtension as pe
import os
from omegaconf import DictConfig


class STACProvider(ABC):
    """Abstract base class for STAC catalog providers."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.catalog = None
        self._initialize_catalog()
    
    @abstractmethod
    def _initialize_catalog(self):
        """Initialize the STAC catalog client."""
        pass
    
    @abstractmethod
    def get_s2_collection_name(self) -> str:
        """Get the Sentinel-2 collection name for this provider."""
        pass
    
    @abstractmethod
    def get_s1_collection_name(self) -> str:
        """Get the Sentinel-1 collection name for this provider."""
        pass
    
    @abstractmethod
    def sign_asset_href(self, href: str) -> str:
        """Sign an asset href for authenticated access."""
        pass
    
    @abstractmethod
    def get_asset_names(self, asset_type: str) -> Dict[str, str]:
        """Get asset names for a given asset type."""
        pass
    
    def search_s2(self,
                  bbox: Tuple[float, float, float, float], 
                  time_of_interest: str, query: dict = None) -> ItemCollection:
        """Search for Sentinel-2 items."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                search = self.catalog.search(
                    collections=[self.get_s2_collection_name()],
                    bbox=bbox,
                    datetime=time_of_interest,
                    query=query
                )
                return search.item_collection()
            except pystac_client.exceptions.APIError as err:
                self.logger.error(f'PySTAC API Error: {err}, {type(err)}')
                if attempt == max_attempts:
                    self.logger.error(f'Maximum number of attempts reached. Exiting.')
                    return ItemCollection([])
                else:
                    self.logger.info(f'Retrying ({attempt}/{max_attempts})...')
            except Exception as err:
                self.logger.error(f'Catalog search failed: {err}, {type(err)}')
                raise err
    
    def search_s1(self,
                  bbox: Tuple[float, float, float, float], 
                  time_of_interest: str,
                  query: dict = None) -> ItemCollection:
        """Search for Sentinel-1 items."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                search = self.catalog.search(
                    collections=[self.get_s1_collection_name()],
                    bbox=bbox,
                    datetime=time_of_interest,
                    query=query
                )
                return search.item_collection()
            except pystac_client.exceptions.APIError as err:
                self.logger.error(f'PySTAC API Error: {err}, {type(err)}')
                if attempt == max_attempts:
                    self.logger.error(f'Maximum number of attempts reached. Exiting.')
                    return ItemCollection([])
                else:
                    self.logger.info(f'Retrying ({attempt}/{max_attempts})...')
            except Exception as err:
                self.logger.error(f'Catalog search failed: {err}, {type(err)}')
                raise err


class MicrosoftPlanetaryComputerProvider(STACProvider):
    """Microsoft Planetary Computer STAC provider implementation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, api_key: Optional[str] = None):
        """Initialize with optional API key parameter."""
        self.api_key = api_key

        os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
        super().__init__(logger)
    
    def _initialize_catalog(self):
        """Initialize the Microsoft Planetary Computer catalog."""
        # Set API key from parameter or environment variable
        if self.api_key:
            os.environ['PC_SDK_SUBSCRIPTION_KEY'] = self.api_key
        elif 'PC_SDK_SUBSCRIPTION_KEY' not in os.environ:
            raise ValueError(
                "Microsoft Planetary Computer API key not found. "
                "Please set the PC_SDK_SUBSCRIPTION_KEY environment variable "
                "or pass api_key parameter. "
                "You can get your API key from https://planetarycomputer.microsoft.com/account/request"
            )
        
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
    
    def get_s2_collection_name(self) -> str:
        return "sentinel-2-l2a"
    
    def get_s1_collection_name(self) -> str:
        return "sentinel-1-rtc"
    
    def sign_asset_href(self, href: str) -> str:
        """Sign asset href using Planetary Computer SDK."""
        return planetary_computer.sign(href)
    
    def get_asset_names(self, asset_type: str) -> Dict[str, str]:
        """Get asset names for Microsoft Planetary Computer."""
        if asset_type == "s2":
            return {
                "visual": "visual",
                "B02": "B02",  # Blue
                "B03": "B03",  # Green
                "B04": "B04",  # Red
                "B08": "B08",  # NIR
                "B11": "B11", # SWIR1
                "B12": "B12", # SWIR2
                "SCL": "SCL",   # Scene Classification Layer
            }
        elif asset_type == "s1":
            return {
                "vv": "vv",
                "vh": "vh"
            }
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")


class AWSProvider(STACProvider):
    """AWS STAC provider implementation.
    
    NOTE: The AWS S1 collection is in a requester pays S3 bucket,
    so cannot be accessed without credentials. Will need to configure
    AWS_SECRET_ACCESS_KEY.
    
    Highly recommend MPC or CDSE over AWS due to problems with Earth Element 84 provider.
    """

    def __init__(self,
                logger: Optional[logging.Logger] = None,
                aws_access_key_id: Optional[str] = None,
                aws_secret_access_key: Optional[str] = None
        ):
        """Set environment variables for rasterio/GDAL use with S3."""
        if aws_access_key_id and aws_secret_access_key:
            os.environ["AWS_REGION"] = "eu-central-1"
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
            os.environ["AWS_HTTPS"] = "YES"
        
        os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
        super().__init__(logger)
    
    def _initialize_catalog(self):
        """Initialize the AWS catalog."""
        # AWS STAC catalog URL - you may need to adjust this
        self.catalog = pystac_client.Client.open(
            "https://earth-search.aws.element84.com/v1"
        )
    
    def get_s2_collection_name(self) -> str:
        return "sentinel-2-l2a"
    
    def get_s1_collection_name(self) -> str:
        # NO RTC ONLY GRD! May need to be careful of this difference.
        return "sentinel-1-grd"
    
    def sign_asset_href(self, href: str) -> str:
        """AWS assets are typically publicly accessible, no signing needed."""
        return href
    
    def get_asset_names(self, asset_type: str) -> Dict[str, str]:
        """Get asset names for AWS."""
        if asset_type == "s2":
            return {
                "visual": "visual",
                "B02": "blue",  # Blue
                "B03": "green",  # Green
                "B04": "red",  # Red
                "B08": "nir",  # NIR
                "B11": "swir16",  # SWIR1
                "B12": "swir22",  # SWIR2
                "SCL": "scl",   # Scene Classification Layer
            }
        elif asset_type == "s1":
            return {
                "vv": "vv",
                "vh": "vh"
            }
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")


class CDSEProvider(STACProvider):
    """Copernicus Data Space Ecosystem STAC provider implementation.
    
    To allow rasterio/GDAL to use S3, set the necessary environment variables
    using the aws_access_key_id and aws_secret_access_key parameters from
    Copernicus. See: https://documentation.dataspace.copernicus.eu/APIs/S3.html#registration.
    """

    def __init__(self,
                logger: Optional[logging.Logger] = None,
                aws_access_key_id: Optional[str] = None,
                aws_secret_access_key: Optional[str] = None
        ):
        """Set environment variables for rasterio/GDAL use with S3."""
        if aws_access_key_id and aws_secret_access_key:
            os.environ["AWS_REGION"] = "default"
            os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
            os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
            os.environ["AWS_HTTPS"] = "YES"
        
        os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
        super().__init__(logger)
    
    def _initialize_catalog(self):
        """Initialize the CDSE catalog."""
        self.catalog = pystac_client.Client.open(
            "https://stac.dataspace.copernicus.eu/v1/"
        )
    
    def get_s2_collection_name(self) -> str:
        return "sentinel-2-l2a"
    
    def get_s1_collection_name(self) -> str:
        return "sentinel-1-grd"
    
    def sign_asset_href(self, href: str) -> str:
        return href
    
    def get_asset_names(self, asset_type: str) -> Dict[str, str]:
        """Get asset names for CDSE."""
        if asset_type == "s2":
            return {
                "visual": "TCI_10m",
                "B02": "B02_10m",  # Blue
                "B03": "B03_10m",  # Green
                "B04": "B04_10m",  # Red
                "B08": "B08_10m",  # NIR
                "B11": "B11_20m", # SWIR1
                "B12": "B12_20m", # SWIR2
                "SCL": "SCL_20m",   # Scene Classification Layer
            }
        elif asset_type == "s1":
            return {
                "vv": "vv",
                "vh": "vh"
            }
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")



def get_stac_provider(provider_name: str,
                mpc_api_key: Optional[str] = None,
                logger: Optional[logging.Logger] = None,
                aws_access_key_id: Optional[str] = None,
                aws_secret_access_key: Optional[str] = None) -> STACProvider:
    f"""Factory function to get the appropriate STAC provider.

    NOTE: AWS Earth Element is not recommended due to partially applied offsets to its products,
    and the fact that the raster extension that encodes whether an offset has been applied is unreliable.
    
    Parameters
    ----------
    provider_name: str
        Name of the provider to use: "mpc" for Microsoft Planetary Computer, "aws" for AWS, "cdse" for Copernicus.
    mpc_api_key: Optional[str]
        API key for Microsoft Planetary Computer.
    aws_access_key_id: Optional[str]
        AWS access key ID (for CDSE).
    aws_secret_access_key: Optional[str]
        AWS secret access key (for CDSE).
    logger : Optional[logging.Logger]
        Logger instance
    
    Returns
    -------
    STACProvider
        Configured STAC provider instance
    """
    
    if provider_name in ["mpc", "microsoft", "planetary_computer"]:
        return MicrosoftPlanetaryComputerProvider(api_key=mpc_api_key, logger=logger)
    elif provider_name in ["aws", "amazon"]:
        return AWSProvider(logger)
    elif provider_name in ["cdse", "copernicus"]:
        return CDSEProvider(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, logger=logger)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported providers: mpc, aws") 