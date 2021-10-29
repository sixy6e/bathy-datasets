from pathlib import Path
from typing import Dict, Any
import json
import structlog
from rasterio.crs import CRS
import pdal
from bathy_datasets import constants

_LOG = structlog.get_logger()


def info(data_uri: str, config_pathname: Path, out_pathname: Path, crs: CRS) -> Dict[str, Any]:
    """Executes the PDAL info pipeline on the TileDB data file."""
    if crs is None:
        crs = constants.DEFAULT_CRS

    info_pipeline = [
        {
            "filename": data_uri,
            "type": "readers.tiledb",
            "override_srs": f"EPSG:{crs.to_epsg()}",
            "config_file": str(config_pathname),
        },
        {"type": "filters.info"},
    ]

    pipeline = pdal.Pipeline(json.dumps(info_pipeline))
    _LOG.info("pdal_filters.info", uri=data_uri)
    _ = pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    return metadata


def stats(data_uri: str, config_pathname: Path, out_pathname: Path, crs: CRS) -> Dict[str, Any]:
    """Executes the PDAL info pipeline on the TileDB data file."""
    if crs is None:
        crs = constants.DEFAULT_CRS

    info_pipeline = [
        {
            "filename": data_uri,
            "type": "readers.tiledb",
            "override_srs": f"EPSG:{crs.to_epsg()}",
            "config_file": str(config_pathname),
        },
        {"type": "filters.stats"},
    ]

    pipeline = pdal.Pipeline(json.dumps(info_pipeline))
    _LOG.info("pdal_filters.stats", uri=data_uri)
    _ = pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    return metadata


def pdal_hexbin(data_uri: str, config_pathname: Path, out_pathname: Path, crs: CRS, edge_length: float) -> Dict[str, Any]:
    """
    Get something akin to a convexhull.
    See:
        https://pdal.io/stages/filters.hexbin.html#filters-hexbin
    """
    if crs is None:
        crs = constants.DEFAULT_CRS

    if edge_length is None:
        edge_length = constants.HEXAGON_EDGE_LENGTH

    hex_pipeline = [
        {
            "filename": data_uri,
            "type": "readers.tiledb",
            "override_srs": f"EPSG:{crs.to_epsg()}",
            "config_file": str(config_pathname),
        },
        {
            "type": "filters.hexbin",
            "edge_size": edge_length,
        },
    ]

    pipeline = pdal.Pipeline(json.dumps(hex_pipeline))
    _LOG.info("pdal_filters.hexbin", uri=data_uri)
    _ = pipeline.execute()
    metadata = json.loads(pipeline.metadata)

    return metadata
