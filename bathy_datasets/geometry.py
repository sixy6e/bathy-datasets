import dask.dataframe
import pandas
import h3
from shapely.geometry import Polygon
import geopandas
from rhealpixdggs import dggs, ellipsoids
from osgeo import osr
import structlog

# from bathy_datasets import constants

_LOG = structlog.get_logger()


def _init_rhealpix():
    """Initialise an rHEALPIX projection using the WGS84 parameters."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(4326)

    ellips = ellipsoids.Ellipsoid(a=crs.GetSemiMajor(), b=crs.GetSemiMinor())
    rhealp = dggs.RHEALPixDGGS(ellips)

    return rhealp


RHEALPIX = _init_rhealpix()


def lon_lat_2_h3(row: pandas.core.series.Series, resolution: int=15):
    """Convert a longitude,latitude pair to a H3 index code."""
    code = h3.geo_to_h3(row.latitude, row.longitude, resolution)
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
    dask_data = dask.dataframe.from_pandas(dataframe.reset_index(), npartitions=npartitions)

    return dask_data.map_partitions(_wrap).compute()


def h3_cell_count(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """Count/tally the unique H3 cells."""
    counts = dataframe.groupby(['h3_index']).h3_index.agg('count').to_frame('count')

    return counts.reset_index()


def h3_cell_geometry_parallel(dataframe: pandas.DataFrame, npartitions: int=2):
    """Return the polygon geometry per H3 cell."""
    def _wrap(dataframe):
        return dataframe.apply((lambda row: h3_cell_geometry(row)), axis=1)

    dask_data = dask.dataframe.from_pandas(dataframe, npartitions=npartitions)

    return dask_data.map_partitions(_wrap).compute()


def dissolve(geodataframe: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Dissovle the H3 geometries. The aim is to simplify the inner geometry."""
    geodataframe["dissolve_field"] = 1
    dissolved = geodataframe.dissolve(by='dissolve_field')

    dissolved.reset_index(drop=True, inplace=True)
    geodataframe.drop("dissolve_field", axis=1, inplace=True)

    return dissolved["geometry"]


def rhealpix_id_geom(row: pandas.core.series.Series, resolution: int=15):
    """Convert a longitude,latitude pair to a rHEALPIX index code and geometry."""
    cell = RHEALPIX.cell_from_point(resolution, (row.longitude, row.latitude), False)

    idx = "".join(map(str, cell.suid))
    polygon = Polygon(cell.vertices(plane=False))

    return idx, polygon


def rhealpix_parallel(dataframe: pandas.DataFrame, npartitions: int=2):
    """Return the rHEALPIX indexes and geometry."""
    def _wrap(dataframe):
        return dataframe.apply((lambda row: rhealpix_id_geom(row)), axis=1)

    dask_data = dask.dataframe.from_pandas(dataframe.reset_index(), npartitions=npartitions)

    return dask_data.map_partitions(_wrap).compute()
