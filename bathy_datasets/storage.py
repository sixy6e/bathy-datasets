import numpy
import tiledb


def mbes_lonlat_domain(tri=False):
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


def ping_beam_domain(
    ping_tile_size=10,
    beam_tile_size=400,
    ping_domain_upper=1_000_000_000,
    beam_domain_upper=1_000_000_000,
):
    """
    Set the array domain using ping and beam numbers as the axes.
    Ideally set the upper domain so that they match the number of elements,
    eg 2179 pings = [0, 2178].
    The beam tile size is ideally set to the number of beams in a ping.
    """
    ping_dim = tiledb.Dim(
        "ping_number",
        domain=(0, ping_domain_upper),
        tile=ping_tile_size,
        dtype=numpy.uint64,
        filters=[
            tiledb.PositiveDeltaFilter(),
            tiledb.RleFilter(),
            tiledb.ZstdFilter(level=16),
        ],
    )

    beam_dim = tiledb.Dim(
        "beam_number",
        domain=(0, beam_domain_upper),
        tile=beam_tile_size,
        dtype=numpy.uint64,
        filters=[
            tiledb.ByteShuffleFilter(),
            tiledb.RleFilter(),
            tiledb.ZstdFilter(level=16),
        ],
    )

    domain = tiledb.Domain(ping_dim, beam_dim)

    return domain


def mbes_attrs(required_attributes=None, include_xy=False):
    """Create the mbes attributes"""
    if required_attributes is None:
        required_attributes = []

    attribs = [
        tiledb.Attr("Z", dtype=numpy.float32, filters=[tiledb.ZstdFilter(level=16)]),
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
            "centre_beam",
            dtype=numpy.uint8,
            filters=[tiledb.RleFilter(), tiledb.ZstdFilter(level=16)],
        ),
        tiledb.Attr("region_code", dtype=str, filters=[tiledb.ZstdFilter(level=16)]),
    ]

    attributes = [at for at in attribs if at.name in required_attributes]

    if include_xy:
        x = tiledb.Attr("X", dtype=numpy.float64, filters=[tiledb.ZstdFilter(level=16)])
        y = tiledb.Attr("Y", dtype=numpy.float64, filters=[tiledb.ZstdFilter(level=16)])
        attributes.insert(0, y)
        attributes.insert(0, x)
    else:
        ping_num = tiledb.Attr(
            "ping_number", dtype=numpy.uint64, filters=[tiledb.ZstdFilter(level=16)]
        )
        beam_num = tiledb.Attr(
            "beam_number", dtype=numpy.uint16, filters=[tiledb.ZstdFilter(level=16)]
        )
        attributes.insert(0, beam_num)
        attributes.insert(0, ping_num)

    return attributes


def create_xy_schema(required_attributes):
    """
    An array schema using X & Y (lon/lat or projected x/y) as the
    dimensional axes.
    The schema is a sparse array, and duplicates are allowed as there is a
    high chance for duplicates when combining all GSFs into a singular
    array due to overlapping passes.
    """
    domain = mbes_lonlat_domain(False)  # only 2 dims for the GMRT project

    attributes = mbes_attrs(required_attributes, include_xy=False)

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


def create_ping_beam_schema(
    required_attributes,
    ping_tile_size=10,
    beam_tile_size=400,
    ping_domain_upper=1_000_000_000,
    beam_domain_upper=1_000_000_000,
):
    """
    Create an array schema using ping and beam as the dimensional axes.
    The ping and beam array schema is dense, just like a 2D grid, and
    we're not allowing duplicates.
    A problem will occur when a beam is suddenly missing. eg all previous
    pings have 100 beams, then all of a sudden we have only 98 beams.
    We'll have isssues in writes as we're expecting to write a full block
    based on the beam domain.
    """
    domain = ping_beam_domain(
        ping_tile_size, beam_tile_size, ping_domain_upper, beam_domain_upper
    )

    attributes = mbes_attrs(required_attributes, include_xy=True)

    schema = tiledb.ArraySchema(
        domain=domain,
        sparse=False,
        attrs=attributes,
        cell_order="row-major",
        tile_order="row-major",
        capacity=100_000,
        allows_duplicates=False,
    )

    return schema


def create_mbes_array(array_uri, schema, ctx=None):
    """Create the TileDB array."""
    with tiledb.scope_ctx(ctx):
        tiledb.Array.create(array_uri, schema)


def append_ping_dataframe(dataframe, array_uri, ctx=None):
    """
    Append the ping dataframe read from a GSF file.
    Only to be used with sparse arrays.
    """
    kwargs = {
        "mode": "append",
        "sparse": True,
        "ctx": ctx,
    }

    tiledb.dataframe_.from_pandas(array_uri, dataframe, **kwargs)
