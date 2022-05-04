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
import attr

# from rhealpixdggs import dggs, ellipsoids  # possibly bring back for production
from shapely.geometry import Polygon


@attr.s(repr=False)
class Ellipsoid:
    """
    At this stage a temporary implementation specifically for a project.
    This is simply reduce the need for a dependency on:
    https://github.com/manaakiwhenua/rhealpixdggs-py

    Afterwards, we should work with TileDB to bundle additional libs
    we require into their build.
    """

    a: float = attr.ib()
    b: float = attr.ib()
    e: float = attr.ib()
    f: float = attr.ib()
    inv_f: float = attr.ib()
    ra: float = attr.ib()

    @classmethod
    def from_crs(cls, crs: CRS = CRS.from_epsg(4326)):
        """Ellipsoid constructor"""
        a = crs.ellipsoid.semi_major_metre
        b = crs.ellipsoid.semi_minor_metre
        e = numpy.sqrt(1 - (b / a) ** 2)
        f = (a - b) / a
        inv_f = 1 / f
        k = numpy.sqrt(
            0.5 * (1 - (1 - e**2) / (2 * e) * numpy.log((1 - e) / (1 + e)))
        )
        ra = a * k

        return cls(a, b, e, f, inv_f, ra)


@attr.s(repr=False)
class RhealpixDGGS:
    """
    At this stage a temporary implementation specifically for a project.
    This is simply reduce the need for a dependency on:
    https://github.com/manaakiwhenua/rhealpixdggs-py

    Afterwards, we should work with TileDB to bundle additional libs
    we require into their build.
    """

    ellipsoid: Ellipsoid = attr.ib()
    n_side: int = attr.ib()
    north_square: int = attr.ib()
    south_square: int = attr.ib()
    max_areal_resolution: int = attr.ib()
    max_resolution: float = attr.ib()

    @classmethod
    def from_ellipsoid(cls, ellipsoid: Ellipsoid = Ellipsoid.from_crs()):
        """RhealpixDGGS constructor."""
        n_side = 3
        north_square = 0
        south_square = 0
        max_areal_resolution = 1
        max_resolution = int(
            numpy.ceil(
                numpy.log(ellipsoid.ra**2 * (2 * numpy.pi / 3) / max_areal_resolution)
                / (2 * numpy.log(n_side))
            )
        )

        return cls(
            ellipsoid,
            n_side,
            north_square,
            south_square,
            max_areal_resolution,
            max_resolution,
        )

    @property
    def cells0(self):
        """Cell ID's at root level"""
        cells = ["N", "O", "P", "Q", "R", "S"]
        return cells

    @property
    def ul_vertex(self):
        """Coordinates for upper left corner of each root level cell."""
        ul_radius_one = {
            self.cells0[0]: numpy.array(
                (-numpy.pi + self.north_square * numpy.pi / 2, 3 * numpy.pi / 4)
            ),
            self.cells0[1]: numpy.array((-numpy.pi, numpy.pi / 4)),
            self.cells0[2]: numpy.array((-numpy.pi / 2, numpy.pi / 4)),
            self.cells0[3]: numpy.array((0, numpy.pi / 4)),
            self.cells0[4]: numpy.array((numpy.pi / 2, numpy.pi / 4)),
            self.cells0[5]: numpy.array(
                (-numpy.pi + self.south_square * numpy.pi / 2, -numpy.pi / 4)
            ),
        }

        # order might have some importance elsewhere
        # ordereddict might be better than looping over the list
        ul_vtx = {cell: self.ellipsoid.ra * ul_radius_one[cell] for cell in self.cells0}

        return ul_vtx

    def cell_width(self, resolution):
        """
        The width of a planar cell.
        For this specific implementation, we're ignoring the case of
        ellipsoidal cells.
        """
        result = self.ellipsoid.ra * (numpy.pi / 2) * self.n_side ** (-resolution)
        return result


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
def _ellipsoidal_shape(region_code: str) -> int:  # , ncodes: int, nside: int) -> int:
    """
    Each cell is one of the following shapes:
        * quad
        * cap
        * dart
        * skew quad

    One of the requirements to determining the North-West vertex.
    The UL in projected rhealpix, may not be the UL in geographics.

    Using the following codes/enum as ID's:
        * 0 -> quad
        * 1 -> cap
        * 2 -> dart
        * 3 -> skew quad

    Keeping this function specific to the nside=3 case. This can be
    expanded at a later date when time permits.
    """
    # TODO; might need to consider turning this into a func that can operate on an array
    #       reason being that when creating geoms this is the place we can remove
    #       vertices from darts

    # quad cells (equatorial)
    # quad_set = set(["O", "P", "Q", "R"])
    quad_set = ["O", "P", "Q", "R"]  # issue with numba < 0.56
    if region_code[0] in quad_set:
        return 0  # quad

    # cap cells
    cap = True
    # if nside % 2 != 1:  ignore for nside=3
    #     cap = False
    for n in region_code[1:]:
        # not going to bother converting string to int
        # if we expand the development of this codebase, then we convert to int's
        if n != "4":  # specific case for nside=3
            cap = False
            break

    if cap:
        return 0

    # dart
    dart = True
    # dart_set = set(["0", "2", "4", "6", "8"])
    dart_set = ["0", "2", "4", "6", "8"]  # issue with numba < 0.56
    for n in region_code[1:]:
        if n not in dart_set:
            dart = False
            break

    if dart:
        return 2

    return 3  # skew quad


@jit(nopython=True)
def _nw_vertex(vertices, cell_shape, s0_code, north_square, south_square, trim_dart, width, authalic_radius):
    """
    Need to cater for rearranging the vertices, as depending on the cell
    shape, the UL projected vertex, may not be the UL in the geographics
    (ellipsoidal) space.  In general, the UL vertex may not be the NW
    vertex in either case.
    """
    if cell_shape <= 1:
        return
    elif cell_shape == 3:  # skew quad
        # calculate the nucleous which is simply the cell centre in planar
        # determine the NW vertex based on which HEALPix triangle the cell is in
        nucleus = (vertices[0, 0] + width / 2.0, vertices[1, 0] - width / 2.0)
        xpt = nucleus[0] / authalic_radius  # scale down
        ypt = nucleus[1] / authalic_radius  # scale down

        # if s0_code not in set(["N", "S"]):  # equatorial (O, P, Q, R)
        if s0_code not in ["N", "S"]:  # equatorial (O, P, Q, R)
            triangle_number = None

        eps = 1e-15

        if s0_code == "N":
            l1 = xpt - (-3 * numpy.pi / 4 + (north_square - 1) * numpy.pi / 2)
            l2 = -xpt + (-3 * numpy.pi / 4 + (north_square + 1) * numpy.pi / 2)
            if ypt < l1 - eps and ypt >= l2 - eps:
                triangle_number = (north_square + 1) % 4
            elif ypt >= l1 - eps and ypt > l2 + eps:
                triangle_number = (north_square + 2) % 4
            elif ypt > l1 + eps and ypt <= l2 + eps:
                triangle_number = (north_square + 3) % 4
            else:
                triangle_number = north_square
        else:
            l1 = xpt - (-3 * numpy.pi / 4 + (south_square + 1) * numpy.pi / 2)
            l2 = -xpt + (-3 * numpy.pi / 4 + (south_square - 1) * numpy.pi / 2)
            if ypt <= l1 + eps and ypt > l2 + eps:
                triangle_number = (south_square + 1) % 4
            elif ypt < l1 - eps and ypt <= l2 + eps:
                triangle_number = (south_square + 2) % 4
            elif ypt >=  l1 - eps and ypt < l2 - eps:
                triangle_number = (south_square + 3) % 4
            else:
                triangle_number = south_square

        if s0_code == "N":
            idx = (triangle_number - north_square) % 4
            idx = -idx
        else:
            idx = (triangle_number - south_square) % 4
    else:  # dart
        # find most poleward value
        idx = numpy.argmax(numpy.abs(vertices[1]))

        if s0_code == "S":
            idx = (idx + 1) % 4

    # X
    verts = list(vertices[0][idx:]) + list(vertices[0][:idx])
    vertices[0] = verts

    # Y
    verts = list(vertices[1][idx:]) + list(vertices[1][:idx])
    vertices[1] = verts

    if trim_dart and cell_shape == 2:
        # nullify
        if s0_code == "N":
            vertices[0, 2] = numpy.nan
            vertices[1, 2] = numpy.nan
        else:
            vertices[0, 1] = numpy.nan
            vertices[1, 1] = numpy.nan


@jit(nopython=True)
def _update_vertices(
    vertices: numpy.ndarray,
    region_codes: numpy.ndarray,
    ncodes: int,
    north_square: int,
    south_square: int,
    trim_dart: bool,
    authalic_radius: float,
    nside: int,
):
    """
    Update the order of the vertices to account for the UL vertex in
    planar space is not necessarily the NW vertex in geographics
    (ellipsoidal) space.
    """
    float_nside = float(nside)
    for i in range(ncodes):
        region_code = str(region_codes[i])
        cell_shape = _ellipsoidal_shape(region_code)
        resolution = len(region_code) - 1
        width = authalic_radius * (numpy.pi / 2) * float_nside ** (-resolution)

        _nw_vertex(
            vertices[:, :, i],
            cell_shape,
            region_code[0],
            north_square,
            south_square,
            trim_dart,
            width,
            authalic_radius,
        )


@jit(nopython=True)
def _rhealpix_code(
    prj_x: numpy.ndarray,
    prj_y: numpy.ndarray,
    resolution: int,
    north_square: int,
    south_square: int,
    authalic_radius: float,
    cell0_width: float,
    ul_vertex: numpy.ndarray,
    nside: int,
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
            y > authalic_radius * numpy.pi / 4
            and y < authalic_radius * 3 * numpy.pi / 4
            and x > authalic_radius * (-numpy.pi + north_square * (numpy.pi / 2))
            and x < authalic_radius * (-numpy.pi / 2 + north_square * (numpy.pi / 2))
        ):
            s0 = cells0[0]
            ul = ul_vertex[0]
        elif (
            y > -authalic_radius * 3 * numpy.pi / 4
            and y < -authalic_radius * numpy.pi / 4
            and x > authalic_radius * (-numpy.pi + south_square * (numpy.pi / 2))
            and x < authalic_radius * (-numpy.pi / 2 + south_square * (numpy.pi / 2))
        ):
            s0 = cells0[5]
            ul = ul_vertex[5]
        elif (
            y >= -authalic_radius * numpy.pi / 4
            and y <= authalic_radius * numpy.pi / 4
            and x >= -authalic_radius * numpy.pi
            and x < -authalic_radius * numpy.pi / 2
        ):
            s0 = cells0[1]
            ul = ul_vertex[1]
        elif (
            y >= -authalic_radius * numpy.pi / 4
            and y <= authalic_radius * numpy.pi / 4
            and x >= -authalic_radius * numpy.pi / 2
            and x < 0
        ):
            s0 = cells0[2]
            ul = ul_vertex[2]
        elif (
            y >= -authalic_radius * numpy.pi / 4
            and y <= authalic_radius * numpy.pi / 4
            and x >= 0
            and x < authalic_radius * numpy.pi / 2
        ):
            s0 = cells0[3]
            ul = ul_vertex[3]
        elif (
            y >= -authalic_radius * numpy.pi / 4
            and y <= authalic_radius * numpy.pi / 4
            and x >= authalic_radius * numpy.pi / 2
            and x < authalic_radius * numpy.pi
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

        # conversion to base(nside) (base(3) in our case)
        # numpy.base_repr didn't work here (within numba) which is fine as we don't
        # need to an additional str -> int conversion for array indexing
        num = abs(int(dy * nside**resolution))
        idx = 0
        if num == 0:
            suid_row[:resolution] = 0
            idx = resolution
        else:
            while num:
                suid_row[idx] = digits[int(num % nside)]
                num //= nside
                idx += 1

        row_ids = suid_row[:resolution][::-1]

        # base conversion
        num = abs(int(dx * nside**resolution))
        idx = 0
        if num == 0:
            suid_col[:resolution] = 0
            idx = resolution
        else:
            while num:
                suid_col[idx] = digits[int(num % nside)]
                num //= nside
                idx += 1

        col_ids = suid_col[:resolution][::-1]

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
    nside: int,
    cell0_width: float,
    authalic_radius: float,
):
    """
    Does the heavy lifting of decoding each region code string identifier.
    """
    boundary = numpy.zeros(
        (2, 4, ncodes), dtype="float64"
    )  # require contiguous blocks later for inverting the coords
    col_map = numpy.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype="uint8")
    row_map = numpy.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype="uint8")

    float_nside = float(nside)  # some calcs result in zero by not casting

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
            dx += nside ** (resolution - res) * suid_col[res - 1]
            dy += nside ** (resolution - res) * suid_row[res - 1]

        dx = dx * float_nside ** (-resolution)
        dy = dy * float_nside ** (-resolution)

        # distance from cell0's origin
        ulx = xy0[0] + cell0_width * dx
        uly = xy0[1] - cell0_width * dy

        width = authalic_radius * (numpy.pi / 2) * float_nside ** (-resolution)

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
    round_coords: bool = True,
    decimals: int = 11,
    planar: bool = False,
    trim_dart: bool = True,
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

    # original setup; possibly bring back for production
    # ellips = ellipsoids.Ellipsoid(
    #     a=to_crs.ellipsoid.semi_major_metre, b=to_crs.ellipsoid.semi_minor_metre
    # )
    # rhealpix = dggs.RHEALPixDGGS(ellips)
    # nside = rhealpix.N_side
    # cell0_width = rhealpix.cell_width(0)
    # authalic_radius = ellips.R_A

    rhealpix = RhealpixDGGS.from_ellipsoid(Ellipsoid.from_crs(to_crs))
    nside = rhealpix.n_side
    cell0_width = rhealpix.cell_width(0)
    authalic_radius = rhealpix.ellipsoid.ra

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
        nside,
        cell0_width,
        authalic_radius,
    )

    if not planar:
        _ = _update_vertices(
            boundary,
            region_codes,
            ncodes,
            rhealpix.north_square,
            rhealpix.south_square,
            trim_dart,
            authalic_radius,
            nside,
        )

        # this next part requires contiguous blocks for inplace calcs
        _ = transformer.transform(
            boundary[0, :, :].ravel(), boundary[1, :, :].ravel(), inplace=True
        )

    # rounding the coordinates as a way of handling differnces in floating point calcs
    # the idea behind this is to enforce (hopefully) neat cell edges
    if round_coords:
        _ = numpy.around(boundary, decimals, out=boundary)

    boundary = boundary.transpose(2, 1, 0)

    if shapely_geometries:
        polygons = []
        for i in range(ncodes):
            # remove any vertices of a dart that have been nullified
            # nullified vertices will only occur if planar=False
            vertices = [v for v in boundary[i] if numpy.isfinite(v[0])]
            polygons.append(Polygon(vertices))

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

    Currently assuming that input lon/lat values are based on EPSG:4326.
    """
    from_crs = CRS.from_epsg(4326)
    to_crs = CRS.from_string("+proj=rhealpix")

    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

    # original setup; possibly bring back for production
    # ellips = ellipsoids.Ellipsoid(
    #     a=from_crs.ellipsoid.semi_major_metre, b=from_crs.ellipsoid.semi_minor_metre
    # )
    # rhealp = dggs.RHEALPixDGGS(ellips)

    rhealp = RhealpixDGGS.from_ellipsoid(Ellipsoid.from_crs(from_crs))

    prj_x, prj_y = transformer.transform(longitude, latitude)

    ns = rhealp.north_square
    ss = rhealp.south_square
    ra = rhealp.ellipsoid.ra
    nside = rhealp.n_side

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
        ra,
        cell0_width,
        ul_vertices,
        nside,
        cells0,
        width_max_resolution,
    )

    return region_codes
