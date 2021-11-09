from pyproj import CRS
from rhealpixdggs import dggs, ellipsoids


def _init_rhealpix():
    """Initialise an rHEALPIX projection using the WGS84 parameters."""
    crs = CRS.from_epsg(4326)
    ellips = ellipsoids.Ellipsoid(
        a=crs.ellipsoid.semi_major_metre, b=crs.ellipsoid.semi_minor_metre
    )
    rhealpix = dggs.RHEALPixDGGS(ellips)

    return rhealpix


# I've been told that all the L2 GSF files will be in geographics
# and most of the time using WGS84
DEFAULT_CRS = CRS.from_epsg(4326)
HEXAGON_EDGE_LENGTH = 0.000139  # 0.5 seconds in length
RHEALPIX = _init_rhealpix()
RHEALPIX_CRS = CRS.from_string("+proj=rhealpix")
