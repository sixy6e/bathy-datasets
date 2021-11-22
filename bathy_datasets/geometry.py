import dask.dataframe
import pandas
import h3
from shapely.geometry import Polygon
import geopandas
from typing import List
import structlog

from bathy_datasets import rhealpix

_LOG = structlog.get_logger()


def h3_code(row: pandas.core.series.Series, resolution: int = 15):
    """Convert a longitude,latitude pair to a H3 ID code."""
    code = h3.geo_to_h3(row.latitude, row.longitude, resolution)
    return code


def h3_cell_geometry(row: pandas.core.series.Series) -> Polygon:
    """Return the geometry for a H3 Cell."""
    cell_geometry = Polygon(h3.h3_to_geo_boundary(row.h3_index, True))

    return cell_geometry


def h3_code_parallel(
    dataframe: pandas.DataFrame, npartitions: int = 2
) -> pandas.core.series.Series:
    """A basic function for generating h3 ID codes in parallel."""

    def _wrap(dataframe):
        return dataframe.apply((lambda row: h3_code(row)), axis=1)

    # dask doesn't like mutli-index dataframes
    dask_data = dask.dataframe.from_pandas(
        dataframe.reset_index(), npartitions=npartitions
    )

    return dask_data.map_partitions(_wrap).compute()


def h3_cell_count(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """Count/tally the unique H3 cells."""
    counts = dataframe.groupby(["h3_index"]).h3_index.agg("count").to_frame("count")

    return counts.reset_index()


def h3_cell_geometry_parallel(dataframe: pandas.DataFrame, npartitions: int = 2):
    """Return the polygon geometry per H3 cell."""

    def _wrap(dataframe):
        return dataframe.apply((lambda row: h3_cell_geometry(row)), axis=1)

    dask_data = dask.dataframe.from_pandas(dataframe, npartitions=npartitions)

    return dask_data.map_partitions(_wrap).compute()


def cell_id_count(dataframe: pandas.DataFrame, col_name: str) -> pandas.DataFrame:
    """
    Count/tally the unique cell ID's to generate coverage density.
    In general this could work for any field
    and was initi.
    """
    counts = dataframe.groupby([col_name])[col_name].agg("count").to_frame("count")
    return counts.reset_index()


def dissolve(geodataframe: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Dissovle the H3 geometries. The aim is to simplify the inner geometry."""
    # geodataframe = geodataframe.copy()
    # geodataframe["dissolve_field"] = 1

    # was finding issues in not all regions were being dissolved
    # could be due to precision of input was very high
    # even though edges were identical (or apparently identical)
    # so buffer by a tiny amount, dissolve then reverse the buffer (erode)
    buffer = geopandas.GeoDataFrame(
        {"geometry": geodataframe.buffer(1e-10, cap_style=3, join_style=2)}
    )
    buffer["dissolve_field"] = 1

    dissolved = buffer.dissolve(by="dissolve_field")
    dissolved.reset_index(drop=True, inplace=True)

    erode = dissolved.buffer(-1e-10, cap_style=3, join_style=2)

    # geodataframe.drop("dissolve_field", axis=1, inplace=True)

    # return dissolved["geometry"]
    return erode


def rhealpix_code(
    dataframe: pandas.core.frame.DataFrame,
    x_name: str = "longitude",
    y_name: str = "latitude",
    resolution: int = 15,
) -> List:
    """Convert a longitude,latitude pair to a rHEALPIX ID code."""
    region_codes = rhealpix.rhealpix_code(
        dataframe[x_name].values, dataframe[y_name].values, resolution
    )

    return region_codes


def rhealpix_code_parallel(
    dataframe: pandas.DataFrame,
    x_name: str = "longitude",
    y_name: str = "latitude",
    npartitions: int = 2,
):
    """Return the rHEALPIX codes."""
    dask_data = dask.dataframe.from_pandas(
        dataframe.reset_index(), npartitions=npartitions
    )

    return dask_data.map_partitions(rhealpix_code, x_name, y_name).compute()


def rhealpix_cell_geometry(
    dataframe: pandas.core.frame.DataFrame, col_name: str
) -> List:
    """Generate rHEALPIX cell geometries for each cell code ID."""
    geometries = rhealpix.rhealpix_geo_boundary(dataframe[col_name].values)

    return geometries


def rhealpix_cell_geometry_parallel(
    dataframe: pandas.DataFrame, col_name: str, npartitions: int = 2
):
    """Generate rHEALPIX cell geometries for each cell code ID."""
    dask_data = dask.dataframe.from_pandas(
        dataframe.reset_index(), npartitions=npartitions
    )

    return dask_data.map_partitions(rhealpix_cell_geometry, col_name).compute()
