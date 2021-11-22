"""
A basic (and perhaps strict) implementation of some functions regarding the
rHEALPIX projection.
The primary requirement was for taking arrays of longitudes and latitudes
and generating rHEALPIX region code identifiers. These could then be used
for creating density maps, and survey coverage in general. The coverage of
each survey can then be neatly nested with other surveys.
The other advantage is we can get a more accurate approximation of area covered
by a survey seeing as rHEALPIX is an equal area projection.

Uses the WGS84 ellipsoid (not the GRS90 spheroid).
Uses 3 sides for a planar cell (3x3 grid per cell)

Most of the code follows the implementation found at:

* https://github.com/manaakiwhenua/rhealpixdggs-py

but reworked to facilitate faster processing by working on arrays.
Some parts could be entirely numpy (or numexpr to save memory)
but instead used numba to retain the simpler per element logic.

The original implementation uses a bunch of lists, tuples, dicts
which are not so friendly within numba (as of 0.53.1), so instead
converted to arrays where possible.
It also provides more functionality, but is classed based working on individual
cells which posed a speed problem. Hence this work.
"""
import numpy
from numba import jit
from pyproj import CRS, Transformer
from rhealpixdggs import dggs, ellipsoids
from shapely.geometry import Polygon


@jit(nopython=True)
def str_to_int(s):
    """
    Work around for converting str to int in numba.
    See https://github.com/numba/numba/issues/5650
    """
    result: int = 0
    final_index: int = len(s) - 1
    for i, v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result


@jit(nopython=True)
def _unpack_res_code(code: str):
    """Given a rHEALPIX code, unpack into a list of integers."""
    unpacked = []
    for i in code[1:]:
        unpacked.append(str_to_int(i))
    return unpacked


@jit(nopython=True)
def _rhealpix_code(
    prj_x: numpy.ndarray,
    prj_y: numpy.ndarray,
    resolution: int,
    north_square: int,
    south_square: int,
    radius: float,
    cell0_width: float,
    ul_vertex: numpy.ndarray,
    nsides: int,
    cells0: numpy.ndarray,
    width_max_resolution: float,
):
    """
    Does the heavy lifting of calculating the region code string identifier.
    """
    region_codes = []
    idx_map = numpy.arange(9).reshape(3, 3)
    digits = numpy.arange(3)

    suid_row = numpy.zeros((15), dtype="uint8")
    suid_col = numpy.zeros((15), dtype="uint8")

    for i in range(prj_x.shape[0]):
        x = prj_x[i]
        y = prj_y[i]

        if (
            y > radius * numpy.pi / 4
            and y < radius * 3 * numpy.pi / 4
            and x > radius * (-numpy.pi + north_square * (numpy.pi / 2))
            and x < radius * (-numpy.pi / 2 + north_square * (numpy.pi / 2))
        ):
            s0 = cells0[0]
            ul = ul_vertex[0]
        elif (
            y > -radius * 3 * numpy.pi / 4
            and y < -radius * numpy.pi / 4
            and x > radius * (-numpy.pi + south_square * (numpy.pi / 2))
            and x < radius * (-numpy.pi / 2 + south_square * (numpy.pi / 2))
        ):
            s0 = cells0[5]
            ul = ul_vertex[5]
        elif (
            y >= -radius * numpy.pi / 4
            and y <= radius * numpy.pi / 4
            and x >= -radius * numpy.pi
            and x < -radius * numpy.pi / 2
        ):
            s0 = cells0[1]
            ul = ul_vertex[1]
        elif (
            y >= -radius * numpy.pi / 4
            and y <= radius * numpy.pi / 4
            and x >= -radius * numpy.pi / 2
            and x < 0
        ):
            s0 = cells0[2]
            ul = ul_vertex[2]
        elif (
            y >= -radius * numpy.pi / 4
            and y <= radius * numpy.pi / 4
            and x >= 0
            and x < radius * numpy.pi / 2
        ):
            s0 = cells0[3]
            ul = ul_vertex[3]
        elif (
            y >= -radius * numpy.pi / 4
            and y <= radius * numpy.pi / 4
            and x >= radius * numpy.pi / 2
            and x < radius * numpy.pi
        ):
            s0 = cells0[4]
            ul = ul_vertex[4]

        dx = abs(x - ul[0]) / cell0_width
        dy = abs(y - ul[1]) / cell0_width

        # the source included a check for delta == 1 (a new cell) which states
        # that it is analytically impossible, but potentially could occur due to
        # floating point calculations
        if dx == 1:
            dx -= 0.5 * width_max_resolution / cell0_width
        if dy == 1:
            dy -= 0.5 * width_max_resolution / cell0_width

        # conversion to base(nsides) (base(3) in our case)
        # numpy.base_repr didn't work here (within numba) which is fine as we don't
        # need to an additional str -> int conversion for array indexing
        num = abs(int(dy * nsides ** resolution))
        idx = 0
        if num == 0:
            suid_row[:resolution] = 0
            idx = resolution
        else:
            while num:
                suid_row[idx] = digits[int(num % nsides)]
                num //= nsides
                idx += 1

        row_ids = suid_row[:idx][::-1]

        # base conversion
        num = abs(int(dx * nsides ** resolution))
        idx = 0
        if num == 0:
            suid_col[:resolution] = 0
            idx = resolution
        else:
            while num:
                suid_col[idx] = digits[int(num % nsides)]
                num //= nsides
                idx += 1

        col_ids = suid_col[:idx][::-1]

        res_codes = []
        for res in range(resolution):
            res_codes.append(idx_map[row_ids[res], col_ids[res]])

        region_codes.append(s0 + "".join([str(val) for val in res_codes]))

    return region_codes


@jit(nopython=True)
def _rhealpix_geo_boundary(
    s0_codes: numpy.ndarray,
    region_codes: numpy.ndarray,
    s0_ul_vertices: numpy.ndarray,
    ncodes: int,
    nsides: int,
    cell0_width: float,
    ellipsoid_radius: float,
):
    """
    Does the heavy lifting of decoding each region code string identifier.
    """
    boundary = numpy.zeros(
        (2, 4, ncodes), dtype="float64"
    )  # require contiguous blocks later for inverting the coords
    col_map = numpy.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype="uint8")
    row_map = numpy.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype="uint8")

    float_nsides = float(nsides)  # some calcs result in zero by not casting

    # ['N', 'O', 'P', 'Q', 'R', 'S'] is the Cell0 order
    for i in range(ncodes):
        s0code = s0_codes[i]
        region_code = str(region_codes[i])
        resolution = len(region_code) - 1

        if s0code == "N":
            xy0 = s0_ul_vertices[0]
        elif s0code == "O":
            xy0 = s0_ul_vertices[1]
        elif s0code == "P":
            xy0 = s0_ul_vertices[2]
        elif s0code == "Q":
            xy0 = s0_ul_vertices[3]
        elif s0code == "R":
            xy0 = s0_ul_vertices[4]
        elif s0code == "S":
            xy0 = s0_ul_vertices[5]

        # it's simpler to map via array indices rather than loop over a dict
        unpacked = numpy.array(_unpack_res_code(region_code), dtype="int64")
        suid_col = col_map[unpacked]
        suid_row = row_map[unpacked]

        # compute the sum of fractional offsets from the origin of each resolution
        dx = 0.0
        dy = 0.0
        for res in range(1, resolution + 1):
            dx += nsides ** (resolution - res) * suid_col[res - 1]
            dy += nsides ** (resolution - res) * suid_row[res - 1]

        dx = dx * float_nsides ** (-resolution)
        dy = dy * float_nsides ** (-resolution)

        # distance from cell0's origin
        ulx = xy0[0] + cell0_width * dx
        uly = xy0[1] - cell0_width * dy

        width = ellipsoid_radius * (numpy.pi / 2) * float_nsides ** (-resolution)

        urx = ulx + width
        ury = uly
        lrx = ulx + width
        lry = uly - width
        llx = ulx
        lly = uly - width

        # the above calcs could be applied directly, but for the moment gives clarity
        boundary[0, 0, i] = ulx
        boundary[0, 1, i] = urx
        boundary[0, 2, i] = lrx
        boundary[0, 3, i] = llx

        boundary[1, 0, i] = uly
        boundary[1, 1, i] = ury
        boundary[1, 2, i] = lry
        boundary[1, 3, i] = lly

    return boundary


def rhealpix_geo_boundary(
    region_codes: numpy.ndarray,
    shapely_geometries: bool = True,
    round_geoms: bool = True,
    decimals: int = 14,
):
    """
    Calculate the RHEALPIX boundary as projected coordinates.
    Most of the code follows the implementation found at:

    * https://github.com/manaakiwhenua/rhealpixdggs-py

    but reworked to facilitate faster processing by working on arrays.
    Some parts could be entirely numpy (or numexpr to save memory)
    but instead used numba to retain the simpler per element logic.
    """
    to_crs = CRS.from_epsg(4326)
    from_crs = CRS.from_string("+proj=rhealpix")

    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

    ellips = ellipsoids.Ellipsoid(
        a=to_crs.ellipsoid.semi_major_metre, b=to_crs.ellipsoid.semi_minor_metre
    )
    rhealpix = dggs.RHEALPixDGGS(ellips)
    nsides = rhealpix.N_side
    cell0_width = rhealpix.cell_width(0)
    ellipsoid_radius = ellips.R_A

    ncodes = region_codes.shape[0]

    # for the case where we have an 'object' datatype
    if "<U" not in region_codes.dtype.name:
        region_codes = region_codes.astype(f"<U{len(region_codes[0])}")

    s0_codes = region_codes.view("<U1")[:: len(region_codes[0])]

    ul_vertices = numpy.zeros((6, 2), "float64")
    for i, cell in enumerate(rhealpix.cells0):
        ul_vertices[i] = rhealpix.ul_vertex[cell]

    boundary = _rhealpix_geo_boundary(
        s0_codes,
        region_codes,
        ul_vertices,
        ncodes,
        nsides,
        cell0_width,
        ellipsoid_radius,
    )

    # this next part requires contiguous blocks for inplace calcs
    _ = transformer.transform(
        boundary[0, :, :].ravel(), boundary[1, :, :].ravel(), inplace=True
    )

    # rounding the coordinates as a way of handling differnces in floating point calcs
    # the idea behind this is to enforce (hopefully) neat cell edges
    if round_geoms:
        _ = numpy.around(boundary, decimals, out=boundary)

    boundary = boundary.transpose(2, 1, 0)

    polygons = []
    for i in range(ncodes):
        polygons.append(Polygon(boundary[i]))

    if shapely_geometries:
        return polygons

    return boundary


def rhealpix_code(longitude, latitude, resolution):
    """
    Given arrays of longitude, latitude and resolution, calculate
    the rHEALPIX region code identifiers.
    Most of the code follows the implementation found at:

    * https://github.com/manaakiwhenua/rhealpixdggs-py

    but reworked to facilitate faster processing by working on arrays.
    Some parts could be entirely numpy (or numexpr to save memory)
    but instead used numba to retain the simpler per element logic.
    """
    from_crs = CRS.from_epsg(4326)
    to_crs = CRS.from_string("+proj=rhealpix")

    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

    ellips = ellipsoids.Ellipsoid(
        a=from_crs.ellipsoid.semi_major_metre, b=from_crs.ellipsoid.semi_minor_metre
    )
    rhealp = dggs.RHEALPixDGGS(ellips)

    prj_x, prj_y = transformer.transform(longitude, latitude)

    ns = rhealp.north_square
    ss = rhealp.south_square
    r = rhealp.ellipsoid.R_A
    nsides = rhealp.N_side

    ul_vertices = numpy.zeros((6, 2), "float64")
    for i, cell in enumerate(rhealp.cells0):
        ul_vertices[i] = rhealp.ul_vertex[cell]

    cell0_width = rhealp.cell_width(0)
    cells0 = numpy.array(rhealp.cells0)
    width_max_resolution = rhealp.cell_width(rhealp.max_resolution)

    region_codes = _rhealpix_code(
        prj_x,
        prj_y,
        resolution,
        ns,
        ss,
        r,
        cell0_width,
        ul_vertices,
        nsides,
        cells0,
        width_max_resolution,
    )

    return region_codes
