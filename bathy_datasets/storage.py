import numpy
import tiledb


def mbes_domain(tri=False):
    """Set array domain."""
    index_filters = tiledb.FilterList([tiledb.ZstdFilter(level=16)])

    xdim = tiledb.Dim(
        "X",
        domain=(None, None),
        tile=1000,
        dtype=numpy.float64,
        filters=index_filters,
    )
    ydim = tiledb.Dim(
        "Y",
        domain=(None, None),
        tile=1000,
        dtype=numpy.float64,
        filters=index_filters,
    )

    if tri:
        # define a third dimension, i.e. depth/z/elevation
        zdim = tiledb.Dim(
            "Z",
            domain=(None, None),
            tile=1000,
            dtype=numpy.float64,
            filters=index_filters,
        )
        domain = tiledb.Domain(xdim, ydim, zdim)
    else:
        domain = tiledb.Domain(xdim, ydim)

    return domain


def mbes_attrs(required_attributes=None):
    """Create the mbes attributes"""
    if required_attributes is None:
        required_attributes = []

    attribs = [
        tiledb.Attr(
            "Z", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "timestamp", dtype="datetime64[ns]", filters=[tiledb.ZstdFilter(level=16)]
        ),  # PDAL doesn't handle native datetimes. if requiring PDAL use numpy.int64
        tiledb.Attr(
            "across_track", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "along_track", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "travel_time", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "beam_angle", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "mean_cal_amplitude",
            dtype=numpy.float32,
            filters=[tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr(
            "beam_angle_forward",
            dtype=numpy.float32,
            filters=[tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr(
            "vertical_error", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "horizontal_error",
            dtype=numpy.float32,
            filters=[tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr(
            "sector_number",
            dtype=numpy.uint8,
            filters=[tiledb.RleFilter(), tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr(
            "beam_flags",
            dtype=numpy.uint8,
            filters=[tiledb.RleFilter(), tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr(
            "ping_flags",
            dtype=numpy.uint8,
            filters=[tiledb.RleFilter(), tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr(
            "tide_corrector", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "depth_corrector",
            dtype=numpy.float32,
            filters=[tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr(
            "heading", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "pitch", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr("roll", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]),
        tiledb.Attr(
            "heave", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "course", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "speed", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "height", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "separation", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "gps_tide_corrector",
            dtype=numpy.float32,
            filters=[tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr(
            "centre_beam", dtype=numpy.uint8, filters=[tiledb.RleFilter(), tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "beam_number", dtype=numpy.uint16, filters=[tiledb.ZstdFilter(level=16)]
        ),
        tiledb.Attr(
            "region_code", dtype=str, filters=[tiledb.ZstdFilter(level=16)]
        ),
    ]

    attributes = [at for at in attribs if at.name in required_attributes]

    return attributes


def mbes_schema(required_attributes):
    """Create the tiledb schema"""
    domain = mbes_domain(False)  # only 2 dims for the project
    attributes = mbes_attrs(required_attributes)

    schema = tiledb.ArraySchema(
        domain=domain,
        sparse=True,
        attrs=attributes,
        cell_order="hilbert",
        tile_order="row-major",
        capacity=100_000,
        allows_duplicates=True,
    )

    return schema


def create_mbes_array(array_uri, required_attributes, ctx=None):
    """Create the TileDB array."""
    schema = mbes_schema(required_attributes)

    with tiledb.scope_ctx(ctx):
        tiledb.Array.create(array_uri, schema)


def append_ping_dataframe(dataframe, array_uri, ctx=None):
    """Append the ping dataframe read from a GSF file."""
    kwargs = {
        "mode": "append",
        "sparse": True,
        "ctx": ctx,
    }

    tiledb.dataframe_.from_pandas(array_uri, dataframe, **kwargs)
