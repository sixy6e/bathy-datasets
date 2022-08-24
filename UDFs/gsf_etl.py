############################## enums.py ##############################
from enum import Enum

# from reap_gsf import reap


class WGS84Coefficients(Enum):
    """
    https://en.wikipedia.org/wiki/Geographic_coordinate_system
    For lonitude and latitude calculations:
        * lat_m_sf = A - B * cos(2 * lat) + C  * cos(4 * lat) - D * cos(6 * lat)
        * lon_m_sf = E * cos(lat) - F * cos(3 * lat) + G * cos(5 * lat)
    """

    A = 111132.92
    B = 559.82
    C = 1.175
    D = 0.0023
    E = 111412.84
    F = 93.5
    G = 0.118


class RecordTypes(Enum):
    """The various record type contained within the GSF file."""

    GSF_HEADER = 1
    GSF_SWATH_BATHYMETRY_PING = 2
    GSF_SOUND_VELOCITY_PROFILE = 3
    GSF_PROCESSING_PARAMETERS = 4
    GSF_SENSOR_PARAMETERS = 5
    GSF_COMMENT = 6
    GSF_HISTORY = 7
    GSF_NAVIGATION_ERROR = 8
    GSF_SWATH_BATHY_SUMMARY = 9
    GSF_SINGLE_BEAM_PING = 10
    GSF_HV_NAVIGATION_ERROR = 11
    GSF_ATTITUDE = 12

    @property
    def func_mapper(self):
        func_map = {
            RecordTypes.GSF_HEADER: read_header,
            RecordTypes.GSF_SWATH_BATHYMETRY_PING: read_bathymetry_ping,
            RecordTypes.GSF_SOUND_VELOCITY_PROFILE: read_svp,
            RecordTypes.GSF_PROCESSING_PARAMETERS: read_processing_parameters,
            RecordTypes.GSF_SENSOR_PARAMETERS: _not_implemented,
            RecordTypes.GSF_COMMENT: read_comment,
            RecordTypes.GSF_HISTORY: read_history,
            RecordTypes.GSF_NAVIGATION_ERROR: _not_implemented,
            RecordTypes.GSF_SWATH_BATHY_SUMMARY: read_swath_bathymetry_summary,
            RecordTypes.GSF_SINGLE_BEAM_PING: _not_implemented,
            RecordTypes.GSF_HV_NAVIGATION_ERROR: _not_implemented,
            RecordTypes.GSF_ATTITUDE: read_attitude,
        }
        return func_map.get(self)


class BeamSubRecordTypes(Enum):
    """The Swath Bathymetry Ping subrecord ID's."""

    DEPTH = 1
    ACROSS_TRACK = 2
    ALONG_TRACK = 3
    TRAVEL_TIME = 4
    BEAM_ANGLE = 5
    MEAN_CAL_AMPLITUDE = 6
    MEAN_REL_AMPLITUDE = 7
    ECHO_WIDTH = 8
    QUALITY_FACTOR = 9
    RECEIVE_HEAVE = 10
    DEPTH_ERROR = 11  # obselete
    ACROSS_TRACK_ERROR = 12  # obselete
    ALONG_TRACK_ERROR = 13  # obselete
    NOMINAL_DEPTH = 14
    QUALITY_FLAGS = 15
    BEAM_FLAGS = 16
    SIGNAL_TO_NOISE = 17
    BEAM_ANGLE_FORWARD = 18
    VERTICAL_ERROR = 19
    HORIZONTAL_ERROR = 20
    INTENSITY_SERIES = 21
    SECTOR_NUMBER = 22
    DETECTION_INFO = 23
    INCIDENT_BEAM_ADJ = 24
    SYSTEM_CLEANING = 25
    DOPPLER_CORRECTION = 26
    SONAR_VERT_UNCERNTAINTY = 27
    SONAR_HORZ_UNCERTAINTY = 28
    DETECTION_WINDOW = 29
    MEAN_ABS_COEF = 30

    @property
    def dtype_mapper(self):
        dtype_map = {
            BeamSubRecordTypes.DEPTH: ">u",
            BeamSubRecordTypes.ACROSS_TRACK: ">i",
            BeamSubRecordTypes.ALONG_TRACK: ">i",
            BeamSubRecordTypes.TRAVEL_TIME: ">u",
            BeamSubRecordTypes.BEAM_ANGLE: ">i",
            BeamSubRecordTypes.MEAN_CAL_AMPLITUDE: ">i",
            BeamSubRecordTypes.MEAN_REL_AMPLITUDE: ">i",
            BeamSubRecordTypes.ECHO_WIDTH: ">u",
            BeamSubRecordTypes.QUALITY_FACTOR: ">u",
            BeamSubRecordTypes.RECEIVE_HEAVE: ">i",
            BeamSubRecordTypes.DEPTH_ERROR: ">u",
            BeamSubRecordTypes.ACROSS_TRACK_ERROR: ">u",
            BeamSubRecordTypes.ALONG_TRACK_ERROR: ">u",
            BeamSubRecordTypes.NOMINAL_DEPTH: ">u",
            BeamSubRecordTypes.QUALITY_FLAGS: ">u",
            BeamSubRecordTypes.BEAM_FLAGS: ">u",
            BeamSubRecordTypes.SIGNAL_TO_NOISE: ">i",
            BeamSubRecordTypes.BEAM_ANGLE_FORWARD: ">u",
            BeamSubRecordTypes.VERTICAL_ERROR: ">u",
            BeamSubRecordTypes.HORIZONTAL_ERROR: ">u",
            BeamSubRecordTypes.INTENSITY_SERIES: ">i",  # not a single type
            BeamSubRecordTypes.SECTOR_NUMBER: ">i",
            BeamSubRecordTypes.DETECTION_INFO: ">i",
            BeamSubRecordTypes.INCIDENT_BEAM_ADJ: ">i",
            BeamSubRecordTypes.SYSTEM_CLEANING: ">i",
            BeamSubRecordTypes.DOPPLER_CORRECTION: ">i",
            BeamSubRecordTypes.SONAR_VERT_UNCERNTAINTY: ">i",  # dtype not defined in 3.09 pdf # noqa: E501
            BeamSubRecordTypes.SONAR_HORZ_UNCERTAINTY: ">i",  # dtype and record not defined in 3.09 pdf # noqa: E501
            BeamSubRecordTypes.DETECTION_WINDOW: ">i",  # dtype and record not defined in 3.09 pdf # noqa: E501
            BeamSubRecordTypes.MEAN_ABS_COEF: ">i",  # dtype and record not defined in 3.09 pdf # noqa: E501
        }
        return dtype_map.get(self)


class SensorSpecific(Enum):
    EM2040 = 149

############################## data_model.py ##############################

import datetime  # type: ignore
from typing import List, Union

import attr
import numpy
import pandas  # type: ignore

# from .enums import RecordTypes


def _dependent_pings(stream, file_record, idx=slice(None)):
    """
    Return a list of dependent pings. This is to aid which pings require scale
    factors from a previous ping.
    """
    results = []
    for i in range(file_record.record_count):
        record = file_record.record(i)
        stream.seek(record.index)
        buffer = stream.read(60)
        subhdr = numpy.frombuffer(buffer[56:], ">i4", count=1)[0]
        subid = (subhdr & 0xFF000000) >> 24
        if subid == 100:
            dep_id = i
            dep = False
        else:
            dep = True
        results.append((dep, dep_id))

    return results[idx]


def _total_ping_beam_count(stream, file_record, idx=slice(None)):
    """
    Return a the total ping beam count.
    The basis for this is that (despite what we were told), the beam count
    can differ between pings. So in order to read slices and insert slices
    into a pre-allocated array, we now need to know the beam count for every
    ping within the slice.
    """
    results = []
    for i in range(file_record.record_count):
        record = file_record.record(i)
        ping_hdr = record.read(stream, None, True)  # read header only
        results.append(ping_hdr.num_beams)

    return numpy.sum(results[idx]), results[idx]


def _ping_dataframe_base(nrows):
    """
    A temporary workaround for the inconsistent schemas that can occur
    between pings.
    For the ARDC project, we're now going to define the attributes to
    be used. If a ping has an additional attribute, it will be ignored, if
    a ping is missing an attribute from the pre-defined set, then null
    values will be used to populate the attribute for the ping.
    Does present a slight disconnect with datatypes being inferred when
    reading a record. Ideally want to avoid any casting. Also requires
    expert input on what the datatypes should be, and the fill value.
    Moving to reap_gsf/enums.py might be better.
    """
    nan = numpy.nan
    dtypes = {
        "X": "float64",
        "Y": "float64",
        "Z": "float32",
        "across_track": "float32",
        "along_track": "float32",
        "beam_angle": "float32",
        "beam_angle_forward": "float32",
        "beam_flags": "uint8",
        "ping_number": "uint64",
        "beam_number": "uint16",
        "centre_beam": "uint8",
        "course": "float32",
        "depth_corrector": "float32",
        "gps_tide_corrector": "float32",
        "heading": "float32",
        "heave": "float32",
        "height": "float32",
        "horizontal_error": "float32",
        # "mean_cal_amplitude": "float32",
        "ping_flags": "uint8",
        "pitch": "float32",
        "roll": "float32",
        "sector_number": "uint8",
        "separation": "float32",
        "speed": "float32",
        "tide_corrector": "float32",
        "timestamp": "datetime64[ns]",
        "travel_time": "float32",
        "vertical_error": "float32",
    }

    fill_value = {
        "X": nan,  # we're in trouble if this is missing
        "Y": nan,  # calculated, so this will be overwritten
        "Z": nan,  # calculated, so this will be overwritten
        "across_track": nan,
        "along_track": nan,
        "beam_angle": nan,
        "beam_angle_forward": nan,
        "beam_flags": 255,
        "ping_number": 0,  # calculated, so this will be overwritten
        "beam_number": 0,  # calculated, so this will be overwritten
        "centre_beam": 0,  # calculated, so this will be overwritten
        "course": nan,
        "depth_corrector": nan,
        "gps_tide_corrector": nan,
        "heading": nan,
        "heave": nan,
        "height": nan,
        "horizontal_error": nan,
        # "mean_cal_amplitude": nan,
        "ping_flags": 255,
        "pitch": nan,
        "roll": nan,
        "sector_number": 0,
        "separation": nan,
        "speed": nan,
        "tide_corrector": nan,
        "timestamp": 0,
        "travel_time": nan,
        "vertical_error": nan,
    }

    ping_dataframe = pandas.DataFrame(
        {
            column: numpy.full((nrows), fill_value[column], dtype=dtypes[column])
            for column in dtypes
        }
    )

    return ping_dataframe


@attr.s(repr=False)
class Record:
    """Instance of a GSF high level record as referenced in RecordTypes."""

    record_type: RecordTypes = attr.ib()
    data_size: int = attr.ib()
    checksum_flag: bool = attr.ib()
    index: int = attr.ib()
    record_index: int = attr.ib()

    def read(self, stream, *args):
        """Read the data associated with this record."""
        stream.seek(self.index)
        data = self.record_type.func_mapper(
            stream, self.data_size, self.checksum_flag, *args
        )
        return data


@attr.s(repr=False)
class FileRecordIndex:

    record_type: RecordTypes = attr.ib()
    record_count: int = attr.ib(init=False)
    data_size: List[int] = attr.ib(repr=False)
    checksum_flag: List[bool] = attr.ib(repr=False)
    indices: List[int] = attr.ib(repr=False)

    def __attrs_post_init__(self):
        self.record_count = len(self.indices)

    def record(self, index):
        result = Record(
            record_type=self.record_type,
            data_size=self.data_size[index],
            checksum_flag=self.checksum_flag[index],
            index=self.indices[index],
            record_index=index,
        )
        return result


@attr.s(repr=False)
class SwathBathymetryPing:
    """
    Data model class for the SwathBathymetryPing sub-records contained
    within a GSF file.
    Essentially all records are combined into a tabular form as a
    pandas.DataFrame construct.
    """

    file_record: FileRecordIndex = attr.ib()
    ping_dataframe: pandas.DataFrame = attr.ib()
    # sensor_dataframe: pandas.DataFrame = attr.ib()

    @classmethod
    def from_records(cls, file_record, stream, idx=slice(None)):
        """Constructor for SwathBathymetryPing. Not supporting idx.step > 1"""
        # TODO testing
        # retrieve the full ping, and a subset.
        # result = full_df[idx.start*nbeams:idx.end*nbeams].reset_index(drop=True) - subs
        # (result.sum() == 0).all() (timestamp won't work, but should be 0 days 00:00:00)
        # ~(result.all()).all() should do the jo
        record_index = list(range(file_record.record_count))
        record_ids = record_index[idx]

        # record dependencies (required for scale factors)
        # only need to resolve the first record as subsequent records are
        # provided with scale_factors
        dependent_pings = _dependent_pings(stream, file_record, idx)

        # get the first record of interest
        if dependent_pings[0][0]:
            rec = file_record.record(dependent_pings[0][1])
            ping_header, scale_factors, df = rec.read(stream)

            rec = file_record.record(record_ids[0])
            ping_header, scale_factors, df = rec.read(stream, scale_factors)
            df["ping_number"] = rec.record_index
        else:
            rec = file_record.record(record_ids[0])
            ping_header, scale_factors, df = rec.read(stream)
            df["ping_number"] = rec.record_index

        # allocating the full dataframe upfront is an attempt to reduce the
        # memory footprint. the append method allocates a whole new copy
        # nrows = file_record.record_count * ping_header.num_beams
        # nrows = len(record_ids) * ping_header.num_beams
        nrows, n_beams = _total_ping_beam_count(stream, file_record, idx)
        # ping_dataframe = pandas.DataFrame(
        #     {
        #         column: numpy.empty((nrows), dtype=df[column].dtype)
        #         for column in df.columns
        #     }
        # )
        ping_dataframe = _ping_dataframe_base(nrows)

        # slices = [
        #     slice(start, start + ping_header.num_beams)
        #     for start in numpy.arange(0, nrows, ping_header.num_beams)
        # ]
        slices = []
        start = 0
        for nbeams in n_beams:
            stop = start + nbeams
            slices.append(slice(start, stop))
            start = stop

        # issues with pandas 1.1.2 and dataframe slicing
        # datatypes are being promoted to higher levels
        # ping_dataframe[slices[0]] = df
        cols = [col for col in ping_dataframe.columns if col in df.columns]
        for col in cols:
            ping_dataframe.loc[slices[0].start : slices[0].stop - 1, col] = df[col].values
            # ping_dataframe.loc[slices[0].start:slices[0].stop-1, col] = df[col]

        for i, rec_id in enumerate(record_ids[1:]):
            rec = file_record.record(rec_id)

            # some pings don't have scale factors and rely on a previous ping
            ping_header, scale_factors, df = rec.read(stream, scale_factors)
            df["ping_number"] = rec.record_index

            # issues with pandas 1.1.2 and dataframe slicing
            # datatypes are being promoted to higher levels
            # ping_dataframe[slices[i + 1]] = df
            cols = [col for col in ping_dataframe.columns if col in df.columns]
            for col in cols:
                ping_dataframe.loc[
                    slices[i + 1].start : slices[i + 1].stop - 1, col
                ] = df[col].values
                # ping_dataframe.loc[slices[i+1].start:slices[i+1].stop-1, col] = df[col]

        return cls(file_record, ping_dataframe)


@attr.s(repr=False)
class PingHeader:
    """
    Data model class for a swath bathymetry ping header record.
    The ping header comes before the ping sub-records that contain
    the beam array for the current ping.
    """

    timestamp: datetime.datetime = attr.ib()
    longitude: float = attr.ib()
    latitude: float = attr.ib()
    num_beams: int = attr.ib()
    center_beam: int = attr.ib()
    ping_flags: int = attr.ib()
    reserved: int = attr.ib()
    tide_corrector: int = attr.ib()
    depth_corrector: int = attr.ib()
    heading: float = attr.ib()
    pitch: float = attr.ib()
    roll: float = attr.ib()
    heave: int = attr.ib()
    course: float = attr.ib()
    speed: float = attr.ib()
    height: int = attr.ib()
    separation: int = attr.ib()
    gps_tide_corrector: int = attr.ib()


@attr.s(repr=False)
class Comment:
    """
    Container for a single comment record.
    """

    timestamp: datetime.datetime = attr.ib()
    comment: str = attr.ib()


@attr.s(repr=False)
class Comments:
    """
    Construct to read and hold all comments within a GSF file.
    """

    file_record: FileRecordIndex = attr.ib()
    comments: Union[List[Comment], None] = attr.ib(default=None)

    @classmethod
    def from_records(cls, file_record, stream):
        """Constructor for all the comments in the GSF file."""
        comments = []
        for i in range(file_record.record_count):
            record = file_record.record(i)
            data = record.read(stream)
            comments.append(data)

        return cls(file_record, comments)


@attr.s(repr=False)
class PingSpatialBounds:

    min_x: Union[float, None] = attr.ib(default=None)
    min_y: Union[float, None] = attr.ib(default=None)
    min_z: Union[float, None] = attr.ib(default=None)
    max_x: Union[float, None] = attr.ib(default=None)
    max_y: Union[float, None] = attr.ib(default=None)
    max_z: Union[float, None] = attr.ib(default=None)

    @classmethod
    def from_dict(cls, bounds):
        """Constructor for PingSpatialBounds."""
        min_x = bounds.get("min_longitude", None)
        min_y = bounds.get("min_latitude", None)
        min_z = bounds.get("min_depth", None)
        max_x = bounds.get("max_longitude", None)
        max_y = bounds.get("max_latitude", None)
        max_z = bounds.get("max_depth", None)

        return cls(min_x, min_y, min_z, max_x, max_y, max_z)


@attr.s(repr=False)
class PingTimestampBounds:
    first_ping: Union[datetime.datetime, None] = attr.ib(default=None)
    last_ping: Union[datetime.datetime, None] = attr.ib(default=None)

    @classmethod
    def from_dict(cls, bounds):
        """Constructor for PingTimestampBounds."""
        first_ping = bounds.get("timestamp_first_ping", None)
        last_ping = bounds.get("timestamp_last_ping", None)

        return cls(first_ping, last_ping)


@attr.s(repr=False)
class SwathBathymetrySummary:
    """
    Container for the swath bathymetry summary record.
    """

    timestamp_bounds: PingTimestampBounds = attr.ib()
    spatial_bounds: PingSpatialBounds = attr.ib()


@attr.s(repr=False)
class Attitude:
    """Data model to combine all attitude records."""

    file_record: FileRecordIndex = attr.ib()
    attitude_dataframe: pandas.DataFrame = attr.ib()

    @classmethod
    def from_records(cls, file_record, stream):
        """Constructor for Attitude."""
        rec = file_record.record(0)
        dataframe = rec.read(stream)

        for i in range(1, file_record.record_count):
            rec = file_record.record(i)
            try:
                dataframe = dataframe.append(rec.read(stream), ignore_index=True)
            except ValueError as err:
                msg = f"record: {rec}, iteration: {i}"
                print(msg, err)
                raise Exception

        dataframe.reset_index(drop=True, inplace=True)

        return cls(file_record, dataframe)


@attr.s(repr=False)
class History:
    """Container for a history record."""

    processing_timestamp: datetime.datetime = attr.ib()
    machine_name: str = attr.ib()
    operator_name: str = attr.ib()
    command: str = attr.ib()
    comment: str = attr.ib()


############################## reap.py ##############################

import datetime
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import attr
import numpy
import pandas

# from .data_model import (
#     Comment,
#     FileRecordIndex,
#     History,
#     PingHeader,
#     PingSpatialBounds,
#     PingTimestampBounds,
#     SwathBathymetrySummary,
# )
# from .enums import BeamSubRecordTypes, RecordTypes, WGS84Coefficients

CHECKSUM_BIT = 0x80000000
NANO_SECONDS_SF = 1e-9
MAX_RECORD_ID = 12
MAX_BEAM_SUBRECORD_ID = 30


def _not_implemented(*args):
    """Handler for records we aren't reading"""
    raise NotImplementedError


def create_datetime(seconds: int, nano_seconds: int) -> datetime.datetime:
    """
    The GSF files store time as a combination of seconds and nano
    seconds past POSIX time.
    """
    timestamp = datetime.datetime.fromtimestamp(
        seconds + NANO_SECONDS_SF * nano_seconds, tz=datetime.timezone.utc
    )
    return timestamp


def record_padding(stream: Union[io.BufferedReader, io.BytesIO]) -> numpy.ndarray:
    """
    GSF requires that all records are multiples of 4 bytes.
    Essentially reads enough bytes so the stream position for the
    record finishes at a multiple of 4 bytes.
    """
    pad = stream.read(stream.tell() % 4)
    return pad


def file_info(
    stream: Union[io.BufferedReader, io.BytesIO], file_size: Optional[int] = None
) -> List[FileRecordIndex]:
    """
    Returns a list of FileRecordIndex objects for each high level record
    type in .enums.RecordTypes.
    The indexes can then be used to quickly traverse through the file.
    """
    # we could be dealing with a gsf stored within a zipfile, or as a cloud object
    if file_size is None:
        fname = Path(stream.name)
        fsize = fname.stat().st_size
    else:
        fsize = file_size

    current_pos = stream.tell()
    stream.seek(0)

    indices: Dict[RecordTypes, List[int]] = {}
    datasize: Dict[RecordTypes, List[int]] = {}
    checksum_flag: Dict[RecordTypes, List[bool]] = {}

    for rtype in RecordTypes:
        indices[rtype] = []
        datasize[rtype] = []
        checksum_flag[rtype] = []

    while stream.tell() < fsize:
        data_size, record_id, flag = read_record_info(stream)

        indices[RecordTypes(record_id)].append(stream.tell())
        datasize[RecordTypes(record_id)].append(data_size)
        checksum_flag[RecordTypes(record_id)].append(flag)

        _ = stream.read(data_size)
        _ = record_padding(stream)

    stream.seek(current_pos)

    r_index = [
        FileRecordIndex(
            record_type=rtype,
            data_size=datasize[rtype],
            checksum_flag=checksum_flag[rtype],
            indices=indices[rtype],
        )
        for rtype in RecordTypes
    ]

    return r_index


def read_record_info(
    stream: Union[io.BufferedReader, io.BytesIO]
) -> Tuple[int, int, bool]:
    """Return the header information for the current record."""
    blob = stream.read(8)
    data_size = numpy.frombuffer(blob, ">u4", count=1)[0]
    record_identifier = numpy.frombuffer(blob[4:], ">i4", count=1)[0]
    checksum_flag = bool(record_identifier & CHECKSUM_BIT)

    return data_size, record_identifier, checksum_flag


def read_header(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, checksum_flag: bool
) -> str:
    """Read the GSF header occuring at the start of the file."""
    blob = stream.read(data_size)
    idx = 0

    if checksum_flag:
        _ = numpy.frombuffer(blob, ">i4", count=1)[0]
        idx += 4

    # TODO; if checksum is read, is data_size - 4 ??
    data = numpy.frombuffer(blob[idx:], f"S{data_size}", count=1)[0]

    _ = record_padding(stream)

    return data


def _proc_param_parser(value: Union[str, datetime.datetime]) -> Any:
    """Convert any strings that have known types such as bools, floats."""
    if isinstance(value, datetime.datetime):  # nothing to do already parsed
        return value

    booleans = {
        "yes": True,
        "no": False,
        "true": True,
        "false": False,
    }

    if "," in value:  # dealing with an array
        array = value.split(",")
        if "." in value:  # assumption on period being a decimal point
            parsed = numpy.array(array, dtype="float").tolist()
        else:
            # could be dealing with an array of "UNKNWN" or "UNKNOWN"
            parsed = ["unknown"] * len(array)
    elif "." in value:  # assumption on period being a decimal point
        parsed = float(value)
    elif value.lower() in booleans:
        parsed = booleans[value.lower()]
    elif value.lower() in ["unknwn", "unknown"]:
        parsed = "unknown"
    else:  # most likely an integer or generic string
        try:
            parsed = int(value)
        except ValueError:
            parsed = value.lower()

    return parsed


def _standardise_proc_param_keys(key: str) -> str:
    """Convert to lowercase, replace any spaces with underscore."""
    return key.lower().replace(" ", "_")


def read_processing_parameters(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, checksum_flag: bool
) -> Dict[str, Any]:
    """
    Read the record containing the parameters used during the data
    processing phase.
    """
    idx = 0

    # blob = stream.readline(data_size)
    blob = stream.read(data_size)

    if checksum_flag:
        _ = numpy.frombuffer(blob, ">i4", count=1)[0]
        idx += 4

    dtype = numpy.dtype(
        [
            ("time_seconds", ">i4"),
            ("time_nano_seconds", ">i4"),
            ("num_params", ">i2"),
        ]
    )
    data = numpy.frombuffer(blob[idx:], dtype, count=1)
    time_seconds = int(data["time_seconds"][0])
    time_nano_seconds = int(data["time_nano_seconds"][0])

    idx += 10

    params: Dict[str, Any] = {}
    for i in range(data["num_params"][0]):
        param_size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
        idx += 2
        data = numpy.frombuffer(blob[idx:], f"S{param_size}", count=1)[0]
        idx += param_size

        key, value = data.decode("utf-8").strip().split("=")

        if key == "REFERENCE TIME":
            value = datetime.datetime.strptime(value, "%Y/%j %H:%M:%S").replace(
                tzinfo=datetime.timezone.utc
            )
            params["processed_datetime"] = value + datetime.timedelta(
                seconds=time_seconds, milliseconds=time_nano_seconds * 1e-6
            )
            continue  # no need to include reference_time

        params[_standardise_proc_param_keys(key)] = _proc_param_parser(value)

    _ = record_padding(stream)

    return params


def read_attitude(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, checksum_flag: bool
) -> pandas.DataFrame:
    """Read an attitude record."""
    blob = stream.read(data_size)
    idx = 0

    if checksum_flag:
        _ = numpy.frombuffer(blob, ">i4", count=1)[0]
        idx += 4

    base_time = numpy.frombuffer(blob[idx:], ">i4", count=2)
    idx += 8

    acq_time = create_datetime(base_time[0], base_time[1])

    num_measurements = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    data: Dict[str, List[Any]] = {
        "timestamp": [],
        "pitch": [],
        "roll": [],
        "heave": [],
        "heading": [],
    }

    dtype = numpy.dtype(
        [
            ("timestamp", ">i2"),
            ("pitch", ">i2"),
            ("roll", ">i2"),
            ("heave", ">i2"),
            ("heading", ">i2"),
        ]
    )
    for _ in range(num_measurements):
        numpy_blob = numpy.frombuffer(blob[idx:], dtype, count=1)[0]
        idx += 10

        data["timestamp"].append(
            acq_time + datetime.timedelta(seconds=numpy_blob["timestamp"] / 1000)
        )
        data["pitch"].append(numpy_blob["pitch"] / 100)
        data["roll"].append(numpy_blob["roll"] / 100)
        data["heave"].append(numpy_blob["heave"] / 100)
        data["heading"].append(numpy_blob["heading"] / 100)

    _ = record_padding(stream)

    dataframe = pandas.DataFrame(data)

    # as these values are stored as 2 byte integers, most precision has already
    # been truncated. therefore convert float64's to float32's
    for column in dataframe.columns:
        if "float" in dataframe.dtypes[column].name:
            dataframe[column] = dataframe[column].values.astype("float32")

    return dataframe


def read_svp(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, flag: bool
) -> pandas.DataFrame:
    """
    Read a sound velocity profile record.
    In the provided samples, the longitude and latitude were both zero.
    It was mentioned that the datetime could be matched (or closely matched)
    with a ping, and the lon/lat could be taken from the ping.
    """
    buffer = stream.read(data_size)
    idx = 0

    dtype = numpy.dtype(
        [
            ("obs_seconds", ">u4"),
            ("obs_nano", ">u4"),
            ("app_seconds", ">u4"),
            ("app_nano", ">u4"),
            ("lon", ">i4"),
            ("lat", ">i4"),
            ("num_points", ">u4"),
        ]
    )

    blob = numpy.frombuffer(buffer, dtype, count=1)
    num_points = blob["num_points"][0]

    idx += 28

    svp = numpy.frombuffer(buffer[idx:], ">u4", count=2 * num_points) / 100
    svp = svp.reshape((num_points, 2))

    data = {
        "longitude": blob["lon"][0] / 10_000_000,
        "latitude": blob["lat"][0] / 10_000_000,
        "depth": svp[:, 0],
        "sound_velocity": svp[:, 1],
        "observation_time": create_datetime(blob["obs_seconds"][0], blob["obs_nano"][0]),
        "applied_time": create_datetime(blob["app_seconds"][0], blob["app_nano"][0]),
    }

    _ = record_padding(stream)

    dataframe = pandas.DataFrame(
        {
            "longitude": data["longitude"] * num_points,
            "latitude": data["latitude"] * num_points,
            "depth": data["depth"] * num_points,
            "sound_velocity": data["sound_velocity"] * num_points,
            "observation_timestamp": [data["observation_time"]] * num_points,
            "applied_timestamp": [data["applied_time"]] * num_points,
        }
    )

    dataframe.rename(
        columns={
            "longitude": "X",
            "latitude": "Y",
        },
        inplace=True,
    )  # using general terms

    return dataframe


def read_swath_bathymetry_summary(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, flag: bool
) -> SwathBathymetrySummary:
    buffer = stream.read(data_size)

    dtype = numpy.dtype(
        [
            ("time_first_ping_seconds", ">i4"),
            ("time_first_ping_nano_seconds", ">i4"),
            ("time_last_ping_seconds", ">i4"),
            ("time_last_ping_nano_seconds", ">i4"),
            ("min_latitude", ">i4"),
            ("min_longitude", ">i4"),
            ("max_latitude", ">i4"),
            ("max_longitude", ">i4"),
            ("min_depth", ">i4"),
            ("max_depth", ">i4"),
        ]
    )

    blob = numpy.frombuffer(buffer, dtype, count=1)

    data = {
        "timestamp_first_ping": create_datetime(
            blob["time_first_ping_seconds"][0], blob["time_first_ping_nano_seconds"][0]
        ),
        "timestamp_last_ping": create_datetime(
            blob["time_last_ping_seconds"][0], blob["time_last_ping_nano_seconds"][0]
        ),
        "min_latitude": blob["min_latitude"][0] / 10_000_000,
        "min_longitude": blob["min_longitude"][0] / 10_000_000,
        "max_latitude": blob["max_latitude"][0] / 10_000_000,
        "max_longitude": blob["max_longitude"][0] / 10_000_000,
        "min_depth": blob["min_depth"][0] / 100,
        "max_depth": blob["max_depth"][0] / 100,
    }

    _ = record_padding(stream)

    time_bounds = PingTimestampBounds.from_dict(data)
    spatial_bounds = PingSpatialBounds.from_dict(data)

    return SwathBathymetrySummary(time_bounds, spatial_bounds)


def read_comment(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, flag: bool
) -> Comment:
    """Read a comment record."""
    dtype = numpy.dtype(
        [
            ("time_comment_seconds", ">i4"),
            ("time_comment_nano_seconds", ">i4"),
            ("comment_length", ">i4"),
        ]
    )
    blob = stream.read(data_size)
    decoded = numpy.frombuffer(blob, dtype, count=1)

    timestamp = create_datetime(
        decoded["time_comment_seconds"][0], decoded["time_comment_nano_seconds"][0]
    )
    the_comment = blob[12:].decode().strip().rstrip("\x00")

    _ = record_padding(stream)

    return Comment(timestamp, the_comment)


def _correct_ping_header(data):
    data_dict = {}

    data_dict["timestamp"] = create_datetime(
        data["time_ping_seconds"][0], data["time_ping_nano_seconds"][0]
    )
    data_dict["longitude"] = float(data["longitude"][0] / 10_000_000)
    data_dict["latitude"] = float(data["latitude"][0] / 10_000_000)
    data_dict["num_beams"] = int(data["number_beams"][0])
    data_dict["center_beam"] = int(data["centre_beam"][0])
    data_dict["ping_flags"] = int(data["ping_flags"][0])
    data_dict["reserved"] = int(data["reserved"][0])
    data_dict["tide_corrector"] = float(data["tide_corrector"][0] / 100)
    data_dict["depth_corrector"] = float(data["depth_corrector"][0] / 100)
    data_dict["heading"] = float(data["heading"][0] / 100)
    data_dict["pitch"] = float(data["pitch"][0] / 100)
    data_dict["roll"] = float(data["roll"][0] / 100)
    data_dict["heave"] = float(data["heave"][0] / 100)
    data_dict["course"] = float(data["course"][0] / 100)
    data_dict["speed"] = float(data["speed"][0] / 100)
    data_dict["height"] = float(data["height"][0] / 1000)
    data_dict["separation"] = float(data["separation"][0] / 1000)
    data_dict["gps_tide_corrector"] = float(data["gps_tide_corrector"][0] / 1000)

    ping_header = PingHeader(**data_dict)

    return ping_header


def _beams_longitude_latitude(
    ping_header: PingHeader, along_track: numpy.ndarray, across_track: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Calculate the longitude and latitude for each beam.

    https://en.wikipedia.org/wiki/Geographic_coordinate_system
    For lonitude and latitude calculations:
        * lat_m_sf = A - B * cos(2 * lat) + C  * cos(4 * lat) - D * cos(6 * lat)
        * lon_m_sf = E * cos(lat) - F * cos(3 * lat) + G * cos(5 * lat)
    """
    # see https://math.stackexchange.com/questions/389942/why-is-it-necessary-to-use-sin-or-cos-to-determine-heading-dead-reckoning # noqa: E501
    lat_radians = math.radians(ping_header.latitude)

    coef_a = WGS84Coefficients.A.value
    coef_b = WGS84Coefficients.B.value
    coef_c = WGS84Coefficients.C.value
    coef_d = WGS84Coefficients.D.value
    coef_e = WGS84Coefficients.E.value
    coef_f = WGS84Coefficients.F.value
    coef_g = WGS84Coefficients.G.value

    lat_mtr_sf = (
        coef_a
        - coef_b * math.cos(2 * lat_radians)
        + coef_c * math.cos(4 * lat_radians)
        - coef_d * math.cos(6 * lat_radians)
    )
    lon_mtr_sf = (
        coef_e * math.cos(lat_radians)
        - coef_f * math.cos(3 * lat_radians)
        + coef_g * math.cos(5 * lat_radians)
    )

    delta_x = math.sin(math.radians(ping_header.heading))
    delta_y = math.cos(math.radians(ping_header.heading))

    lon2 = (
        ping_header.longitude
        + delta_y / lon_mtr_sf * across_track
        + delta_x / lon_mtr_sf * along_track
    )
    lat2 = (
        ping_header.latitude
        - delta_x / lat_mtr_sf * across_track
        + delta_y / lat_mtr_sf * along_track
    )

    return lon2, lat2


def _ping_dataframe(
    ping_header: PingHeader, subrecords: Dict[BeamSubRecordTypes, numpy.ndarray]
) -> pandas.DataFrame:
    """Construct the dataframe for the given ping."""
    # convert beam arrays to point cloud structure (i.e. generate coords for every beam)
    longitude, latitude = _beams_longitude_latitude(
        ping_header,
        subrecords[BeamSubRecordTypes.ALONG_TRACK],
        subrecords[BeamSubRecordTypes.ACROSS_TRACK],
    )

    dataframe = pandas.DataFrame({k.name.lower(): v for k, v in subrecords.items()})
    dataframe.insert(0, "latitude", latitude)
    dataframe.insert(0, "longitude", longitude)

    # include the header info in the dataframe as that was desired by many in the survey
    ignore = [
        "longitude",
        "latitude",
        "num_beams",
        "reserved",
        "center_beam",
    ]
    for key, value in attr.asdict(ping_header).items():
        if key in ignore:
            continue
        if key == "timestamp":
            value = value.replace(tzinfo=None)
        dataframe[key] = value

    # most perpendicular beam
    dataframe["centre_beam"] = False
    if ping_header.ping_flags == 0:
        query = dataframe.beam_flags == 0
        subset = dataframe[query]
        if subset.shape[0]:
            # require suitable beams from this ping to determine the centre beam
            idx = subset.across_track.abs().idxmin()
            dataframe.loc[idx, "centre_beam"] = True

    # beam number
    dataframe["beam_number"] = numpy.arange(ping_header.num_beams).astype("uint16")

    # float32 conversion;
    # it seems all the attributes have had some level of truncation applied
    # thus losing some level of precision

    ignore = ["longitude", "latitude"]
    for column in dataframe.columns:
        if column in ignore:
            continue
        if "float" in dataframe.dtypes[column].name:
            dataframe[column] = dataframe[column].values.astype("float32")

    dataframe["ping_flags"] = dataframe["ping_flags"].values.astype("uint8")
    dataframe["beam_flags"] = dataframe["beam_flags"].values.astype("uint8")
    dataframe["centre_beam"] = dataframe["centre_beam"].values.astype("uint8")

    dataframe.rename(
        columns={
            "longitude": "X",
            "latitude": "Y",
            "depth": "Z",
        },
        inplace=True,
    )  # using general terms

    return dataframe


def _ping_scale_factors(
    num_factors: int, buffer: str, idx: int
) -> Tuple[Dict[BeamSubRecordTypes, numpy.ndarray], int]:
    """Small util for populating the ping scale factors."""
    scale_factors: Dict[BeamSubRecordTypes, numpy.ndarray] = {}

    for i in range(num_factors):
        blob = numpy.frombuffer(buffer[idx:], ">i4", count=3)
        beam_subid = (blob[0] & 0xFF000000) >> 24
        _ = (blob & 0x00FF0000) >> 16  # compression flag

        scale_factors[BeamSubRecordTypes(beam_subid)] = blob[1:]
        idx = idx + 12

    return scale_factors, idx


def _ping_beam_subrecord(
    ping_header: PingHeader,
    buffer: str,
    scale_factors: Dict[BeamSubRecordTypes, numpy.ndarray],
    subrecord_size: int,
    subrecord_id: int,
    idx: int,
) -> Tuple[numpy.ndarray, int, int, int]:
    """Small util for reading and converting a ping beam subrecord."""
    size = subrecord_size // ping_header.num_beams
    sub_rec_type = BeamSubRecordTypes(subrecord_id)
    dtype = f"{sub_rec_type.dtype_mapper}{size}"
    sub_rec_blob = numpy.frombuffer(buffer[idx:], dtype, count=ping_header.num_beams)

    idx = idx + size * ping_header.num_beams

    scale = scale_factors[sub_rec_type][0]
    offset = scale_factors[sub_rec_type][1]

    data = sub_rec_blob / scale - offset

    subrecord_hdr = numpy.frombuffer(buffer[idx:], ">i4", count=1)[0]
    idx = idx + 4

    subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
    subrecord_size = subrecord_hdr & 0x00FFFFFF

    return data, subrecord_id, subrecord_size, idx


def read_bathymetry_ping(
    stream,
    data_size,
    flag,
    scale_factors=None,
    header_only=False,
) -> Tuple[PingHeader, Dict[BeamSubRecordTypes, numpy.ndarray], pandas.DataFrame]:
    """Read and digest a bathymetry ping record."""
    idx = 0
    blob = stream.read(data_size)

    dtype = numpy.dtype(
        [
            ("time_ping_seconds", ">i4"),
            ("time_ping_nano_seconds", ">i4"),
            ("longitude", ">i4"),
            ("latitude", ">i4"),
            ("number_beams", ">i2"),
            ("centre_beam", ">i2"),
            ("ping_flags", ">i2"),
            ("reserved", ">i2"),
            ("tide_corrector", ">i2"),
            ("depth_corrector", ">i4"),
            ("heading", ">u2"),
            ("pitch", ">i2"),
            ("roll", ">i2"),
            ("heave", ">i2"),
            ("course", ">u2"),
            ("speed", ">u2"),
            ("height", ">i4"),
            ("separation", ">i4"),
            ("gps_tide_corrector", ">i4"),
        ]
    )

    ping_header = _correct_ping_header(numpy.frombuffer(blob, dtype=dtype, count=1))

    if header_only:
        return ping_header

    idx += 56  # includes 2 bytes of spare space

    # first subrecord
    subrecord_hdr = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
    subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
    subrecord_size = subrecord_hdr & 0x00FFFFFF

    idx += 4

    if subrecord_id == 100:
        # scale factor subrecord
        num_factors = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
        idx += 4

        # if we have input sf's return new ones
        # some pings don't store a scale factor record and rely on
        # ones read from a previous ping
        scale_factors, idx = _ping_scale_factors(num_factors, blob, idx)
    else:
        if scale_factors is None:
            # can't really do anything sane
            # could return the unscaled data, but that's not the point here
            raise Exception("Record has no scale factors")

        # roll back the index by 4 bytes
        idx -= 4

    subrecord_hdr = numpy.frombuffer(blob[idx:], ">i4", count=1)[0]
    idx += 4

    subrecord_id = (subrecord_hdr & 0xFF000000) >> 24
    subrecord_size = subrecord_hdr & 0x00FFFFFF

    # beam array subrecords
    subrecords = {}
    while subrecord_id <= MAX_BEAM_SUBRECORD_ID and subrecord_id > 0:
        data, new_subrecord_id, subrecord_size, idx = _ping_beam_subrecord(
            ping_header, blob, scale_factors, subrecord_size, subrecord_id, idx
        )
        subrecords[BeamSubRecordTypes(subrecord_id)] = data
        subrecord_id = new_subrecord_id

    dataframe = _ping_dataframe(ping_header, subrecords)

    # SKIPPING:
    #     * sensor specific sub records
    #     * intensity series
    # print(idx)

    return ping_header, scale_factors, dataframe


def read_history(
    stream: Union[io.BufferedReader, io.BytesIO], data_size: int, flag: bool
):
    """Read a history record."""
    blob = stream.read(data_size)
    idx = 0

    time = numpy.frombuffer(blob, ">i4", count=2)
    timestamp = create_datetime(time[0], time[1])
    idx += 8

    size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    end_idx = idx + size + 1
    machine_name = blob[idx:end_idx].decode().strip().rstrip("\x00")
    idx += size

    size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    end_idx = idx + size + 1
    operator_name = blob[idx:end_idx].decode().strip().rstrip("\x00")
    idx += size

    size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    end_idx = idx + size + 1
    command = blob[idx:end_idx].decode().strip().rstrip("\x00")
    idx += size

    size = numpy.frombuffer(blob[idx:], ">i2", count=1)[0]
    idx += 2

    end_idx = idx + size + 1
    comment = blob[idx:end_idx].decode().strip().rstrip("\x00")

    history = History(timestamp, machine_name, operator_name, command, comment)

    return history


def dependent_pings(
    stream: Union[io.BufferedReader, io.BytesIO], file_record: FileRecordIndex
) -> List[Tuple[bool, int]]:
    """
    Generate a list of tuple's indicating whether a SwathBathymetryPing record
    depends on a prior SwathBathymetryPing record. This is useful for
    paralising the read of bathymetry pings, where scale factors may not be
    present in some records.
    """
    results = []
    for i in range(file_record.record_count):
        record = file_record.record(i)
        stream.seek(record.index)
        buffer = stream.read(60)
        subhdr = numpy.frombuffer(buffer[56:], ">i4", count=1)[0]
        subid = (subhdr & 0xFF000000) >> 24

        if subid == 100:
            dep_id = i
            dep = False
        else:
            dep = True

        results.append((dep, dep_id))

    return results


############################## rhealpix.py ##############################

import numpy
from numba import jit

# from pyproj import CRS, Transformer  # not yet avail on UDF exec build
from rasterio.crs import CRS
from rasterio import warp
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
    # def from_crs(cls, crs: CRS = CRS.from_epsg(4326)):
    def from_crs(cls):
        """Ellipsoid constructor"""
        # a = crs.ellipsoid.semi_major_metre  # not yet avail on UDF exec build
        # b = crs.ellipsoid.semi_minor_metre  # not yet avail on UDF exec build
        a = 6378137.0
        b = 6356752.314245179
        e = numpy.sqrt(1 - (b / a) ** 2)
        f = (a - b) / a
        inv_f = 1 / f
        k = numpy.sqrt(0.5 * (1 - (1 - e**2) / (2 * e) * numpy.log((1 - e) / (1 + e))))
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

        suid_row[:] = 0
        suid_col[:] = 0

    return region_codes


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
    # from_crs = CRS.from_epsg(4326)
    from_crs = CRS.from_wkt(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    )
    to_crs = CRS.from_string("+proj=rhealpix")

    # temporary until pyproj avail on UDF exec build
    # transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

    # original setup; possibly bring back for production
    # ellips = ellipsoids.Ellipsoid(
    #     a=from_crs.ellipsoid.semi_major_metre, b=from_crs.ellipsoid.semi_minor_metre
    # )
    # rhealp = dggs.RHEALPixDGGS(ellips)

    # rhealp = RhealpixDGGS.from_ellipsoid(Ellipsoid.from_crs(from_crs))
    rhealp = RhealpixDGGS.from_ellipsoid(Ellipsoid.from_crs())

    # prj_x, prj_y = transformer.transform(longitude, latitude)
    prj_x, prj_y = warp.transform(from_crs, to_crs, longitude, latitude)
    prj_x = numpy.array(prj_x)
    prj_y = numpy.array(prj_y)

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


############################## <not-defined>.py ##############################

from pathlib import Path
import boto3
import json
import urllib
import s3fs
import tiledb.cloud


class Encoder(json.JSONEncoder):
    """Extensible encoder to handle non-json types."""

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)

        return super(Encoder, self).default(obj)


def retrieve_stream(uri, access_key, skey):
    """
    Not testing the creation of the stream object at this point.
    But for testing, we also need to keep the download to occur only
    once.
    """
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=skey)
    dev_resource = session.resource("s3")
    uri = urllib.parse.urlparse(uri)
    obj = dev_resource.Object(bucket_name=uri.netloc, key=uri.path[1:])
    stream = io.BytesIO(obj.get()["Body"].read())

    return stream, obj.content_length


def decode_gsf(args, idx=None):
    stream, length = args
    finfo = file_info(stream, length)
    sbp = SwathBathymetryPing.from_records(finfo[1], stream, idx)
    sbp.ping_dataframe["region_code"] = rhealpix_code(
        sbp.ping_dataframe.X, sbp.ping_dataframe.Y, 15
    )
    return sbp.ping_dataframe, finfo


def write_gsf_info(gsf_uri, access_key, skey):
    """
    Write the GSF file info (record types and counts) as a JSON document.
    """
    stream, stream_length = retrieve_stream(gsf_uri, access_key, skey)
    finfo = file_info(stream, stream_length)
    # finfo_dict = {"record_count": {rt.record_type.name: rt.record_count for rt in finfo}}
    finfo_dict = {
        "gsf_uri": gsf_uri,
        "size": stream_length,
        "file_record_types": {},
    }
    for frtype in finfo:
        finfo_dict["file_record_types"][frtype.record_type.name] = attr.asdict(frtype)
        finfo_dict["file_record_types"][frtype.record_type.name][
            "record_type"
        ] = frtype.record_type.value

    # cater for empty GSF files (the why is being looked into)
    if finfo[1].record_count:
        ping_hdr, ping_sf, ping_df = finfo[1].record(0).read(stream)
        finfo_dict["bathymetry_ping_schema"] = list(ping_df.columns)
    else:
        finfo_dict["bathymetry_ping_schema"] = []

    input_uri = urllib.parse.urlparse(gsf_uri)
    pth = Path(input_uri.path)
    output_uri = urllib.parse.ParseResult(
        input_uri.scheme,
        input_uri.netloc,
        pth.with_suffix(".json").as_posix(),
        "",
        "",
        "",
    )

    # fs = s3fs.S3FileSystem(key=access_key, secret=skey)

    # remove if path exists
    # if fs.exists(output_uri.geturl()):
    #     fs.rm(output_uri.geturl())

    # with fs.open(output_uri.geturl(), "w") as out_ds:
    #     out_ds.write(json.dumps(finfo_dict))
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=skey)
    s3_client = session.client("s3")

    # remove if object exists
    try:
        s3_client.get_object(Bucket=output_uri.netloc, Key=output_uri.path[1:])
    except s3_client.exceptions.NoSuchKey:
        pass
    else:
        s3_client.delete_object(Bucket=output_uri.netloc, Key=output_uri.path[1:])

    json_data = json.dumps(finfo_dict, cls=Encoder)
    s3_client.put_object(
        Body=json_data, Bucket=output_uri.netloc, Key=output_uri.path[1:]
    )


def write_gsf_info_list(gsf_uris, access_key, skey):
    """
    Process a list of GSF files and write the GSF file info as a JSON document.
    """
    for gsf_uri in gsf_uris:
        write_gsf_info(gsf_uri, access_key, skey)


def append_ping_dataframe(dataframe, array_uri, access_key, skey, chunks=100000):
    """Append the ping dataframe read from a GSF file."""
    config = tiledb.Config(
        {"vfs.s3.aws_access_key_id": access_key, "vfs.s3.aws_secret_access_key": skey}
    )
    ctx = tiledb.Ctx(config=config)
    kwargs = {
        "mode": "append",
        "sparse": True,
        "ctx": ctx,
    }

    idxs = [
        (start, start + chunks) for start in numpy.arange(0, dataframe.shape[0], chunks)
    ]

    for idx in idxs:
        subset = dataframe[idx[0]:idx[1]]
        tiledb.dataframe_.from_pandas(array_uri, subset, **kwargs)


def write_ping_beam_dims_dataframe(dataframe, array_uri, access_key, skey):
    """
    Write the ping dataframe to a TileDB using a dense ping-beam dimensional
    axes.
    """
    config = tiledb.Config(
        {"vfs.s3.aws_access_key_id": access_key, "vfs.s3.aws_secret_access_key": skey}
    )
    ctx = tiledb.Ctx(config=config)

    ping_start_idx = int(dataframe.ping_number.min())
    ping_end_idx = int(dataframe.ping_number.max()) + 1

    col_info = tiledb.dataframe_._get_column_infos(dataframe, None, None)
    data_dict, _nullna = tiledb.dataframe_._df_to_np_arrays(dataframe, col_info, None)

    with tiledb.open(array_uri, "w", ctx=ctx) as ds:
        ds[ping_start_idx:ping_end_idx, :] = data_dict


def ingest_gsf_slice(
    file_record,
    stream,
    access_key,
    skey,
    array_uri,
    idx=slice(None),
    cell_frequency=False,
    ping_beam_dims=False,
):
    """
    General steps:
    Extract the ping data.
    Calculate the rHEALPIX code.
    Summarise the rHEALPIX codes (frequency count).
    Get timestamps of first and last pings.
    Write the ping data to a TileDB array.
    res = [df.groupby(["key"])["key"].agg("count").to_frame("count").reset_index() for i in range(3)]
    df2 = pandas.concat(res)
    df2.groupby(["key"])["count"].agg("sum")
    """
    swath_pings = SwathBathymetryPing.from_records(file_record, stream, idx)
    swath_pings.ping_dataframe["region_code"] = rhealpix_code(
        swath_pings.ping_dataframe.X, swath_pings.ping_dataframe.Y, 15
    )

    # frequency of dggs cells
    if cell_frequency:
        cell_count = (
            swath_pings.ping_dataframe.groupby(["region_code"])["region_code"]
            .agg("count")
            .to_frame("count")
            .reset_index()
        )

        start_end_time = [
            swath_pings.ping_dataframe.timestamp.min().to_pydatetime(),
            swath_pings.ping_dataframe.timestamp.max().to_pydatetime(),
        ]

    else:
        cell_count = None
        start_end_time = None

    # write to tiledb array
    if ping_beam_dims:
        write_ping_beam_dims_dataframe(swath_pings.ping_dataframe, array_uri, access_key, skey)
    else:
        append_ping_dataframe(swath_pings.ping_dataframe, array_uri, access_key, skey)

    return cell_count, start_end_time


def ingest_gsf_slices(gsf_uri, access_key, skey, array_uri, slices, cell_frequency=False, ping_beam_dims=False):
    """
    Ingest a list of ping slices from a given GSF file.
    """
    stream, stream_length = retrieve_stream(gsf_uri, access_key, skey)
    finfo = file_info(stream, stream_length)
    ping_file_record = finfo[1]

    cell_counts = []
    start_end_timestamps = []

    for idx in slices:
        count, start_end_time = ingest_gsf_slice(
            ping_file_record, stream, access_key, skey, array_uri, idx, cell_frequency, ping_beam_dims
        )
        cell_counts.append(count)
        start_end_timestamps.append(start_end_time)

    if cell_frequency:
        # aggreate the ping slices and calculate the cell counts
        concatenated = pandas.concat(cell_counts)
        cell_count = (
            concatenated.groupby(["region_code"])["count"]
            .agg("sum")
            .to_frame("count")
            .reset_index()
        )

        # aggregate the min and max timestamps, then find the min max timestamps
        timestamps_df = pandas.DataFrame(
            {
                "start_datetime": [i[0] for i in start_end_timestamps],
                "end_datetime": [i[1] for i in start_end_timestamps],
            }
        )

        start_end_timestamp = [
            timestamps_df.start_datetime.min().to_pydatetime(),
            timestamps_df.end_datetime.max().to_pydatetime(),
        ]

    else:
        cell_count = None
        start_end_timestamp = None

    return cell_count, start_end_timestamp


token = ""

tiledb.cloud.login(token=token)

tiledb.cloud.udf.register_generic_udf(decode_gsf, name="decode_gsf", namespace="AusSeabed")
tiledb.cloud.udf.register_generic_udf(
    retrieve_stream, name="retrieve_stream", namespace="AusSeabed"
)
tiledb.cloud.udf.register_generic_udf(
    write_gsf_info, name="write_gsf_info", namespace="AusSeabed"
)
tiledb.cloud.udf.register_generic_udf(
    write_gsf_info_list, name="write_gsf_info_list", namespace="AusSeabed"
)
tiledb.cloud.udf.register_generic_udf(
    ingest_gsf_slices, name="ingest_gsf_slices", namespace="AusSeabed"
)
