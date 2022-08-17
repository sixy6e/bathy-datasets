"""
General purpose utilities.
Initially a placeholeder for a bunch of general purpose funcs developed
as part of the ARDC-GMRT project.
"""

import datetime
from pathlib import Path
import json
import attr
import numpy
from numba import jit
import pandas
import tiledb

from typing import Any, List, Tuple, Union

from reap_gsf import reap


def reduce_region_codes(region_codes: pandas.DataFrame) -> pandas.DataFrame:
    """
    The reduce part of the map-reduce construct for handling the region_code counts.
    Combine all the region_code counts then summarise the results.
    """
    df = pandas.concat(region_codes)
    cell_count = (
        df.groupby(["region_code"])["count"].agg("sum").to_frame("count").reset_index()
    )

    return cell_count


def reduce_timestamps(
    timestamps: List[List[datetime.datetime]],
) -> List[datetime.datetime]:
    """
    The reduce part of the map-reduce construct for handling the region_code counts.
    Combine all the region_code counts then summarise the results.
    """
    timestamps_df = pandas.DataFrame(
        {
            "start_datetime": [i[0] for i in timestamps],
            "end_datetime": [i[1] for i in timestamps],
        }
    )
    start_end_timestamp = [
        timestamps_df.start_datetime.min().to_pydatetime(),
        timestamps_df.end_datetime.max().to_pydatetime(),
    ]

    return start_end_timestamp


def scatter(iterable, n) -> List[Any]:
    """
    Evenly scatters an interable by `n` blocks.
    Sourced from:
    http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts

    :param iterable:
        An iterable or preferably a 1D list or array.

    :param n:
        An integer indicating how many blocks to create.

    :return:
        A `list` consisting of `n` blocks of roughly equal size, each
        containing elements from `iterable`.
    """
    q, r = len(iterable) // n, len(iterable) % n
    res = (iterable[i * q + min(i, r) : (i + 1) * q + min(i + 1, r)] for i in range(n))

    return list(res)


def start_end_timestamps(
    array_uri: str, access_key: str, skey: str
) -> List[datetime.datetime]:
    """
    Find the min/max of the timestamp attribute.
    """
    config = tiledb.Config(
        {"vfs.s3.aws_access_key_id": access_key, "vfs.s3.aws_secret_access_key": skey}
    )
    ctx = tiledb.Ctx(config=config)

    with tiledb.open(array_uri, ctx=ctx) as ds:
        query = ds.query(attrs=["timestamp"], coords=False)
        df = query.df[:]

    start_end_time = [
        df.timestamp.min().to_pydatetime(),
        df.timestamp.max().to_pydatetime(),
    ]

    return start_end_time


def reduce_resolution(
    df: pandas.DataFrame, resolution=12, chunks=10000
) -> pandas.DataFrame:
    """
    Reduce the rHEALPix cell code resolutions listed in a pandas.DataFrame
    in a chunked fashion.
    """

    def reduce_res(dataframe: pandas.DataFrame, resolution: int) -> pandas.DataFrame:
        res = resolution + 1
        reduced = pandas.DataFrame(
            {
                "region_code": dataframe.region_code.str[0:res],
                "count": dataframe["count"].values,
            }
        )

        return reduced

    def group_res(dataframe: pandas.DataFrame) -> pandas.DataFrame:
        return (
            dataframe.groupby(["region_code"])["count"]
            .agg("sum")
            .to_frame("count")
            .reset_index()
        )

    idxs = [(start, start + chunks) for start in numpy.arange(0, df.shape[0], chunks)]
    idx0 = idxs[0]
    subset = df[idx0[0] : idx0[1]]
    base_reduced = group_res(reduce_res(subset, resolution))

    for idx in idxs[1:]:
        subset = df[idx[0] : idx[1]]
        reduced = reduce_res(subset, resolution)
        concatenated = pandas.concat([base_reduced, reduced], copy=False)
        base_reduced = group_res(concatenated)

    return base_reduced


def write_chunked(
    df: pandas.DataFrame, out_uri: str, ctx: tiledb.Ctx, chunks: int = 10000
) -> None:
    """
    Write a dataframe to a dense tiledb array in a chunked fashion.
    """
    idxs = [(start, start + chunks) for start in numpy.arange(0, df.shape[0], chunks)]
    rows_written = 0
    kwargs = {
        "sparse": False,
        "column_types": {"region_code": str, "count": numpy.uint64},
        "ctx": ctx,
    }
    for idx in idxs:
        subset = df[idx[0] : idx[1]]
        kwargs["row_start_idx"] = rows_written
        tiledb.dataframe_.from_pandas(out_uri, subset, **kwargs)
        kwargs["mode"] = "append"
        rows_written += len(subset)


@jit(nopython=True)
def strtoint(s):
    return ord(s) - 48


@jit(nopython=True)
def _unpack_code(region_codes: numpy.ndarray, ncodes, res):
    """
    The workhorse for unpacking an array of rHEALPix code identifiers.
    Basic implementation for resolutions [0, 15].
    """
    resolutions = [str(f"R{i}") for i in range(res)]
    unpacked = {
        "R1": numpy.zeros(ncodes, dtype="uint8"),
        "R2": numpy.zeros(ncodes, dtype="uint8"),
        "R3": numpy.zeros(ncodes, dtype="uint8"),
        "R4": numpy.zeros(ncodes, dtype="uint8"),
        "R5": numpy.zeros(ncodes, dtype="uint8"),
        "R6": numpy.zeros(ncodes, dtype="uint8"),
        "R7": numpy.zeros(ncodes, dtype="uint8"),
        "R8": numpy.zeros(ncodes, dtype="uint8"),
        "R9": numpy.zeros(ncodes, dtype="uint8"),
        "R10": numpy.zeros(ncodes, dtype="uint8"),
        "R11": numpy.zeros(ncodes, dtype="uint8"),
        "R12": numpy.zeros(ncodes, dtype="uint8"),
        "R13": numpy.zeros(ncodes, dtype="uint8"),
        "R14": numpy.zeros(ncodes, dtype="uint8"),
        "R15": numpy.zeros(ncodes, dtype="uint8"),
    }

    r0 = numpy.zeros(ncodes, dtype="<U1")

    for i in range(ncodes):
        code = str(region_codes[i])
        r0[i] = code[0]

        for j in range(1, res):
            unpacked[resolutions[j]][i] = strtoint(code[j])

    return r0, unpacked


def unpack_code(region_codes: numpy.ndarray, dataframe=True):
    """
    Unpacking an array of rHEALPix code identifiers into separate columns
    of a dataframe.
    """
    res = len(region_codes[0])
    region_codes = region_codes.astype(f"<U{len(region_codes[0])}")
    r0, unpacked = _unpack_code(region_codes, region_codes.shape[0], res)
    unpacked_dict = {"R0": r0}

    for key in unpacked:
        unpacked_dict[key] = unpacked[key]
    if dataframe:
        result = pandas.DataFrame(unpacked_dict)
    else:
        result = unpacked_dict

    return result


def write_sparse_rhealpix_chunked(
    df: pandas.DataFrame, out_uri: str, ctx: tiledb.Ctx, chunks: int = 10000
) -> None:
    """
    Write a dataframe to a sparse tiledb array using rHEALPix as the
    coordinate dimensions of the array. Data is written in a chunked
    fashion.
    Requires the output array to have already been created.
    """
    idxs = [(start, start + chunks) for start in numpy.arange(0, len(df), chunks)]
    kwargs = {
        "mode": "append",
        "sparse": True,
        "ctx": ctx,
    }

    for idx in idxs:
        subset = df[idx[0] : idx[1]]
        new_df = unpack_code(subset.region_code.values)
        new_df["region_code"] = subset.region_code.values
        new_df["count"] = subset["count"].values

        tiledb.dataframe_.from_pandas(out_uri, new_df, **kwargs)


def filter_empty_files(
    files: List[str], vfs: tiledb.vfs.VFS
) -> Tuple[List[str], List[str]]:
    """
    Filter out GSF's containing no Pings so we don't attempt to process them.
    """
    empty_files = []
    non_empty_files = []

    for pathname in files:
        metadata_pathname = pathname.replace(".gsf", ".json")

        with vfs.open(metadata_pathname) as src:
            gsf_metadata = json.loads(src.read())

        ping_count = gsf_metadata["file_record_types"]["GSF_SWATH_BATHYMETRY_PING"][
            "record_count"
        ]

        if ping_count == 0:
            empty_files.append(pathname)
        else:
            non_empty_files.append(pathname)

    return non_empty_files, empty_files


def filter_large_files(
    files: List[str], size_limit_mb: int, vfs: tiledb.vfs.VFS
) -> Tuple[List[str], List[str]]:
    """
    Filter out GSF's that are large than size_limit_mb so that they're processed locally.
    Probably not neeeded anymore as we can stream the GSF files using tiledb's virtual file
    system (VFS).
    """
    manageable_files = []
    large_files = []

    for pathname in files:
        metadata_pathname = pathname.replace(".gsf", ".json")

        with vfs.open(metadata_pathname) as src:
            gsf_metadata = json.loads(src.read())

        if (gsf_metadata["size"] / 1024 / 1024) > size_limit_mb:
            large_files.append(pathname)
        else:
            manageable_files.append(pathname)

    return manageable_files, large_files


def cell_frequency(
    array_uri: str, cell_frequency_uri: str, access_key: str, skey: str
) -> None:
    """
    Calculate the frequency distirbution of each region code (cell count).
    Result is written to a tiledb array.
    """
    config = tiledb.Config(
        {"vfs.s3.aws_access_key_id": access_key, "vfs.s3.aws_secret_access_key": skey}
    )

    ctx = tiledb.Ctx(config=config)
    kwargs = {
        "sparse": False,
        "column_types": {
            "region_code": str,
            "count": numpy.uint64,
        },
        "ctx": ctx,
    }

    with tiledb.open(array_uri, ctx=ctx) as ds:
        query = ds.query(attrs=["region_code"], coords=False)
        df = query.df[:]

    frequency_df = (
        df.groupby(["region_code"])["region_code"]
        .agg("count")
        .to_frame("count")
        .reset_index()
    )

    tiledb.dataframe_.from_pandas(cell_frequency_uri, frequency_df, **kwargs)


def load_and_concat(
    array_uris: List[str], ctx: tiledb.Ctx, out_uri: Union[str, None] = None
):
    """
    Given a list of uri's pointing TileDB arrays of cell frequency counts,
    combine and groupby the cell ids to establish new frequency counts.
    Not ideal to do all files at once. Instead use an iterative divide and
    conquer approach.
    Eg 100 files can be split into 10 groups of 10, then 2 groups of 5,
    then finally one group of two.
    """

    def concat(array_uris: List[str], ctx: tiledb.Ctx) -> pandas.DataFrame:
        """
        Concatenate multiple arrays into a singluar dataframe.
        Attempting to do this in a minimalistic memory fashion. But there's
        a lot that goes on under the hood in pandas.
        We could pre-allocate an array and read everything into it.
        We could also apply this iteratively.
        """
        data = []

        for uri in array_uris:
            with tiledb.open(uri, ctx=ctx) as ds:
                data.append(ds.df[:])

        concatenated = pandas.concat(data, copy=False)

        return concatenated

    concatenated = concat(array_uris, ctx)
    summarised = (
        concatenated.groupby(["region_code"])["count"]
        .agg("sum")
        .to_frame("count")
        .reset_index()
    )

    if out_uri:
        write_chunked(summarised, out_uri, ctx, chunks=1000000)
    else:
        return summarised


class Encoder(json.JSONEncoder):
    """Extensible encoder to handle non-json types."""

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)

        return super(Encoder, self).default(obj)


def write_gsf_info(gsf_uri: str, vfs: tiledb.vfs.VFS) -> None:
    """
    Write the GSF file info (record types and counts) as a JSON document.
    Also acts as a kind of file index to the GSF enabling specific records
    to be read via a byte range request.
    """
    with vfs.open(gsf_uri) as stream:
        stream_length = stream._nbytes
        finfo = reap.file_info(stream, stream_length)

        finfo_dict = {
            "gsf_uri": gsf_uri,
            "size": stream_length,
            "file_record_types": {},
        }

        # gather each file record type
        for frtype in finfo:
            finfo_dict["file_record_types"][frtype.record_type.name] = attr.asdict(frtype)
            finfo_dict["file_record_types"][frtype.record_type.name][
                "record_type"
            ] = frtype.record_type.value

        # cater for empty GSF files (the why they're empty is to be looked into)
        if finfo[1].record_count:
            ping_hdr, ping_sf, ping_df = finfo[1].record(0).read(stream)
            finfo_dict["bathymetry_ping_schema"] = list(ping_df.columns)
        else:
            finfo_dict["bathymetry_ping_schema"] = []

    output_uri = gsf_uri.replace(".gsf", ".json")

    json_data = json.dumps(finfo_dict, cls=Encoder)

    with vfs.open(output_uri, "wb") as src:
        src.write(json_data)
