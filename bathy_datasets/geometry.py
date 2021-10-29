import dask
import pandas
import h3
from shapely.geometry import Polygon
import structlog

from bathy_datasets import constants

_LOG = structlog.get_logger()


def lon_lat_2_h3(row: pandas.core.series.Series, resolution: int=15):
    """Convert a longitude,latitude pair to a H3 index code."""
    code = h3.geo_to_h3(row.Y, row.X, resolution)
    return code


def h3_cell_geometry(row: pandas.core.series.Series) -> Polygon:
    """Return the geometry for a H3 Cell."""
    cell_geometry = Polygon(h3.h3_to_geo_boundary(row.h3_index, True))

    return cell_geometry


def h3_index_parallel(dataframe: pandas.DataFrame, npartitions: int=2) -> pandas.core.series.Series:
    """A basic function for generating h3 index codes in parallel."""
    def _wrap(dataframe):
        return dataframe.apply((lambda row: lon_lat_2_h3(row)), axis=1)

    # dask doesn't like mutli-index dataframes
    dask_data = dask.dataframe(dataframe.reset_index(), npartitions=npartitions)

    return dask_data.mpa_partitions(_wrap).compute()


def h3_cell_count(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """Count/tally the unique H3 cells."""
    counts = dataframe.groupby(['h3_index']).h3_index.agg('count').to_frame('count')

    return counts.reset_index()


def h3_cell_geometry_parallel(dataframe: pandas.DataFrame, npartitions: int=2):
    """Return the polygon geometry per H3 cell."""
    def _wrap(dataframe):
        return dataframe.apply((lambda row: h3_cell_geometry(row)), axis=1)

    dask_data = dask.dataframe(dataframe, npartitions=npartitions)

    return dask_data.mpa_partitions(_wrap).compute()


def dissolve(geodataframe: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Dissovle the H3 geometries. The aim is to simplify the inner geometry."""
    geodataframe["dissolve_field"] = 1
    dissolved = geodataframe.dissolve(by='dissolve_field')

    dissolved.reset_index(drop=True, inplace=True
    geodataframe.drop("dissolve_field", axis=1, inplace=True)

    return dissolved["geometry"]
