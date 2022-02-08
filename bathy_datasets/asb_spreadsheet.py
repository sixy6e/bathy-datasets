import dateutil
from enum import Enum
from typing import List, Dict, Any
import pandas
import structlog


_LOG = structlog.get_logger()


class SurveyColumns(Enum):
    """Field and value labels for survey sheets"""

    FIELD = "Survey/Mission Attributes"
    VALUE = "User entered (some defaults pre-entered)"


class BathyColumns(Enum):
    """Field and value labels for bathymetric sheets"""

    FIELD = "Bathy Metadata Attributes"
    VALUE = "User entered (some defaults pre-entered)"


SHEET_NAMES: List = [
    "Survey (General)",
    "Survey (Citation)",
    "Survey (Details)",
    "Survey (Technical)",
    "Bathymetry (General)",
    "Bathymetry (Citation)",
    "Bathymetry (Details)",
    "Bathymetry (Technical)",
]
# SURVEY_COLUMNS: List = [
#     "Survey/Mission Attributes",
#     "User entered (some defaults pre-entered)",
# ]
# BATHY_COLUMNS: List = [
#     "Bathy Metadata Attributes",
#     "User entered (some defaults pre-entered)",
# ]
EXCLUDE: str = "Inherited"
# VALUE_COLUMN: str = "User entered (some defaults pre-entered)"
HEADER: int = 1


def read_sheet(
    pathname: str, sheetname: str, storage_options: Dict[str, Any] = None
) -> pandas.DataFrame:
    """Read a specific sheet from an ASB Excel Spreadsheet."""
    dataframe = pandas.read_excel(
        pathname, sheet_name=sheetname, header=HEADER, storage_options=storage_options
    )

    return dataframe


def standardise_name(name):
    """
    Standardise field names: Survey (Title) -> survery_title
    """
    result = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    # remove any starting and ending "_" that have been inserted
    start_loc = 1 if result[0] == "_" else 0
    loc = result.rfind("_")
    end_loc = loc if loc == (len(result) - 1) else len(result)

    return result[start_loc:end_loc]


def clean_metadata(dataframe, column_type):
    """Cleanup the metadata; names, values. Combine duplicates etc."""
    dupes = dataframe[column_type.FIELD.value].duplicated()
    non_dupe_fields = dataframe[~dupes][column_type.FIELD.value].values

    metadata = {name: [] for name in non_dupe_fields}

    for idx, row in dataframe.iterrows():
        metadata[row[column_type.FIELD.value]].append(row[column_type.VALUE.value])

    clean_md = {}
    for key, value in metadata.items():
        new_value = value if len(value) > 1 else value[0]
        if "keyword" in key.lower():
            clean_md["keywords"] = new_value.split(",")
        else:
            if "\u2026" in key:
                # remove any horizonal ellipsis, all cases have a proceeding "_"
                new_key = standardise_name(key[key.find("\u2026") + 2:])
            else:
                new_key = standardise_name(key)
            if "datetime" in new_key:
                # we're getting a mixture of everything
                # hopefully dateutil can resolve most of it
                new_value = dateutil.parser.parse(new_value)
            clean_md[new_key] = new_value

    return clean_md


def harvest(
    pathname: str, storage_options: Dict[str, Any] = None
) -> Dict[str, pandas.DataFrame]:
    """
    Harvest metadata from the AusSeabed Spreadsheet.
    storage_options passes through to s3fs.
    See https://s3fs.readthedocs.io/en/latest/index.html for more info.
    eg {"profile":"pl019-data"}
    """
    metadata: Dict[str, pandas.DataFrame] = {}

    for sheet_name in SHEET_NAMES:
        enumerator = BathyColumns if "bath" in sheet_name.lower() else SurveyColumns
        cols = [column.value for column in enumerator]
        dataframe = read_sheet(pathname, sheet_name, storage_options)

        query = (dataframe.Requirement != EXCLUDE) & (
            dataframe[enumerator.VALUE.value].notnull()
        )
        filtered = dataframe[query]
        subset = filtered[cols].reset_index(drop=True)

        metadata[sheet_name] = clean_metadata(subset, enumerator)

    cleaned = {standardise_name(key): value for key, value in metadata.items()}

    return cleaned
