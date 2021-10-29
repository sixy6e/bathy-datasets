from rasterio.crs import CRS


# I've been told that all the L2 GSF files will be in geographics
# and most of the time using WGS84
DEFAULT_CRS = CRS.from_epsg(4326)
HEXAGON_EDGE_LENGTH = 0.000139  # 0.5 seconds in length
