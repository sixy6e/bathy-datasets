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
def _unpack_res_code(code):
    """Given a rHEALPIX code, unpack into a list of integers."""
    unpacked = []
    for i in code[1:]:
        unpacked.append(str_to_int(i))
    return unpacked


@jit(nopython=True)
def _rhealpix_code(prj_x, prj_y, resolution, ns, ss, r, width, ul_vertex, n, cells0):
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
            y > r * numpy.pi / 4
            and y < r * 3 * numpy.pi / 4
            and x > r * (-numpy.pi + ns * (numpy.pi / 2))
            and x < r * (-numpy.pi / 2 + ns * (numpy.pi / 2))
        ):
            s0 = cells0[0]
            ul = ul_vertex[0]
        elif (
            y > -r * 3 * numpy.pi / 4
            and y < -r * numpy.pi / 4
            and x > r * (-numpy.pi + ss * (numpy.pi / 2))
            and x < r * (-numpy.pi / 2 + ss * (numpy.pi / 2))
        ):
            s0 = cells0[5]
            ul = ul_vertex[5]
        elif (
            y >= -r * numpy.pi / 4
            and y <= r * numpy.pi / 4
            and x >= -r * numpy.pi
            and x < -r * numpy.pi / 2
        ):
            s0 = cells0[1]
            ul = ul_vertex[1]
        elif (
            y >= -r * numpy.pi / 4
            and y <= r * numpy.pi / 4
            and x >= -r * numpy.pi / 2
            and x < 0
        ):
            s0 = cells0[2]
            ul = ul_vertex[2]
        elif (
            y >= -r * numpy.pi / 4
            and y <= r * numpy.pi / 4
            and x >= 0
            and x < r * numpy.pi / 2
        ):
            s0 = cells0[3]
            ul = ul_vertex[3]
        elif (
            y >= -r * numpy.pi / 4
            and y <= r * numpy.pi / 4
            and x >= r * numpy.pi / 2
            and x < r * numpy.pi
        ):
            s0 = cells0[4]
            ul = ul_vertex[4]

        dx = abs(x - ul[0]) / width
        dy = abs(y - ul[1]) / width

        num = abs(int(dy * n ** resolution))
        idx = 0
        if num == 0:
            suid_row[:resolution] = 0
            idx = resolution
        else:
            while num:
                suid_row[idx] = digits[int(num % n)]
                num //= n
                idx += 1

        row_ids = suid_row[:idx][::-1]

        num = abs(int(dx * n ** resolution))
        idx = 0
        if num == 0:
            suid_col[:resolution] = 0
            idx = resolution
        else:
            while num:
                suid_col[idx] = digits[int(num % n)]
                num //= n
                idx += 1

        col_ids = suid_col[:idx][::-1]

        res_codes = []
        for res in range(resolution):
            res_codes.append(idx_map[row_ids[res], col_ids[res]])

        region_codes.append(s0 + "".join([str(val) for val in res_codes]))

    return region_codes


@jit(nopython=True)
def _rhealpix_geo_boundary(
    s0_codes,
    region_codes,
    s0_ul_vertices,
    ncodes,
    nsides,
    cell0_width,
    ellipsoid_radius,
):
    """
    Does the heavy lifting of decoding each region code string identifier.
    """
    boundary = numpy.zeros(
        (2, 4, ncodes), dtype="float64"
    )  # require contiguous blocks later
    col_map = numpy.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype="uint8")
    row_map = numpy.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype="uint8")

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

        dx = 0.0
        dy = 0.0
        for res in range(1, resolution + 1):
            dx += nsides ** (resolution - res) * suid_col[res - 1]
            dy += nsides ** (resolution - res) * suid_row[res - 1]

        dx = dx * float(nsides) ** (-resolution)
        dy = dy * float(nsides) ** (-resolution)

        ulx = xy0[0] + cell0_width * dx
        uly = xy0[1] - cell0_width * dy

        width = ellipsoid_radius * (numpy.pi / 2) * nsides ** (-resolution)

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


def rhealpix_geo_boundary(region_codes):
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
    boundary = boundary.transpose(2, 1, 0)

    polygons = []
    for i in range(ncodes):
        polygons.append(Polygon(boundary[i]))

    return polygons


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
    rhealpix = dggs.RHEALPixDGGS(ellips)

    prj_x, prj_y = transformer.transform(longitude, latitude)

    ns = rhealpix.north_square
    ss = rhealpix.south_square
    r = rhealpix.ellipsoid.R_A
    nsides = rhealpix.N_side

    ul_vertices = numpy.zeros((6, 2), "float64")
    for i, cell in enumerate(rhealpix.cells0):
        ul_vertices[i] = rhealpix.ul_vertex[cell]

    cell0_width = rhealpix.cell_width(0)
    cells0 = numpy.array(rhealpix.cells0)

    region_codes = _rhealpix_code(
        prj_x, prj_y, resolution, ns, ss, r, cell0_width, ul_vertices, nsides, cells0
    )

    return region_codes
