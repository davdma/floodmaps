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
    
    def _initialize_catalog(self):
        """Initialize the Microsoft Planetary Computer catalog."""
        # Set API key if not already set - will need to configure this for your own account
        if 'PC_SDK_SUBSCRIPTION_KEY' not in os.environ:
            os.environ['PC_SDK_SUBSCRIPTION_KEY'] = 'a613baefa08445269838bc3bc0dfe2d9'
        
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
                "SCL": "SCL"   # Scene Classification Layer
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
    AWS_SECRET_ACCESS_KEY."""
    
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
                "SCL": "scl"   # Scene Classification Layer
            }
        elif asset_type == "s1":
            return {
                "vv": "vv",
                "vh": "vh"
            }
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")


def get_stac_provider(provider_name: str, logger: Optional[logging.Logger] = None) -> STACProvider:
    """Factory function to get the appropriate STAC provider."""
    provider_name = provider_name.lower()
    
    if provider_name in ["mpc", "microsoft", "planetary_computer"]:
        return MicrosoftPlanetaryComputerProvider(logger)
    elif provider_name in ["aws", "amazon"]:
        return AWSProvider(logger)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported providers: mpc, aws, copernicus") 