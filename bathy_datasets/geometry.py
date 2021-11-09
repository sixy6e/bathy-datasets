import dask.dataframe
import pandas
import h3
from shapely.geometry import Polygon
import geopandas
import structlog

from bathy_datasets import constants, rhealpix

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


def dissolve(geodataframe: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Dissovle the H3 geometries. The aim is to simplify the inner geometry."""
    geodataframe["dissolve_field"] = 1
    dissolved = geodataframe.dissolve(by="dissolve_field")

    dissolved.reset_index(drop=True, inplace=True)
    geodataframe.drop("dissolve_field", axis=1, inplace=True)

    return dissolved["geometry"]


def rhealpix_code(
    dataframe: pandas.core.frame.DataFrame,
    x_name: str = "longitude",
    y_name: str = "latitude",
    resolution: int = 15,
) -> pandas.core.series.Series:
    """Convert a longitude,latitude pair to a rHEALPIX ID code."""
    region_codes = rhealpix.rhealpix_code(
        dataframe[x_name].values, dataframe[y_name].values, resolution
    )

    return pandas.Series(region_codes)


def rhealpix_code_parallel(dataframe: pandas.DataFrame, npartitions: int = 2):
    """Return the rHEALPIX codes."""

    def _wrap(dataframe):
        return dataframe.apply(rhealpix_code, axis=1)

    dask_data = dask.dataframe.from_pandas(
        dataframe.reset_index(), npartitions=npartitions
    )

    return dask_data.map_partitions(_wrap).compute()


def rhealpix_cell_geometry(
    dataframe: pandas.core.frame.DataFrame, col_name: str
) -> pandas.core.series.Series:
    """Generate rHEALPIX cell geometries for each cell code ID."""
    geometries = rhealpix.rhealpix_geo_boundary(dataframe[col_name].values)

    return pandas.Series(geometries)


def rhealpix_cell_geometry_parallel(dataframe: pandas.DataFrame, npartitions: int = 2):
    """Generate rHEALPIX cell geometries for each cell code ID."""

    def _wrap(dataframe):
        return dataframe.apply(rhealpix_code, axis=1)

    dask_data = dask.dataframe.from_pandas(
        dataframe.reset_index(), npartitions=npartitions
    )

    return dask_data.map_partitions(_wrap).compute()
