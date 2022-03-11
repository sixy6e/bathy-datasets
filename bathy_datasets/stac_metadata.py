from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List
import uuid
import urllib.parse

import fiona
from fiona.session import AWSSession
import numpy

import pystac
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.pointcloud import (
    PointcloudExtension,
    SchemaType,
    PhenomenologyType,
    Schema,
    Statistic,
)
from pystac.extensions.scientific import ScientificExtension
import s3fs
import tiledb
import tiledb.cloud


class Encoder(json.JSONEncoder):
    """Extensible encoder to handle non-json types."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)

        return super(Encoder, self).default(obj)


def tiledb_context(access_key: Union[str, None], secret_key: Union[str, None]):
    """
    Simple func for creating a tiledb context for handling
    AWS security.
    """
    if access_key is None:
        ctx = None
    else:
        config = tiledb.Config(
            {
                "vfs.s3.aws_access_key_id": access_key,
                "vfs.s3.aws_secret_access_key": secret_key,
            }
        )
        ctx = tiledb.Ctx(config=config)

    return ctx


def create_pdal_schema(
    uri: str, access_key: Union[str, None] = None, secret_key: Union[str, None] = None
):
    """
    Create a PDAL-like schema structure from the TileDB array.
    """
    ctx = tiledb_context(access_key, secret_key)

    # STAC.PC uses the terms, signed, unsigned, floating
    dtype_mapper = {
        "i": "signed",
        "u": "unsigned",
        "f": "floating",
    }

    ds = tiledb.open(uri, ctx=ctx)
    domain = [ds.schema.domain.dim(i) for i in range(ds.schema.ndim)]
    attributes = [ds.schema.attr(i) for i in range(ds.schema.nattr)]

    pdal_schema = []

    for dim in domain:
        pdal_schema.append(
            {
                "name": dim.name,
                "size": dim.dtype.itemsize,
                "type": dtype_mapper[dim.dtype.kind],
            }
        )

    for attrib in attributes:
        pdal_schema.append(
            {
                "name": attrib.name,
                "size": attrib.dtype.itemsize,
                "type": dtype_mapper[attrib.dtype.kind],
            }
        )

    return pdal_schema


def data_geometry(
    uri: str, access_key: Union[str, None] = None, secret_key: Union[str, None] = None
) -> Dict[str, Any]:
    """
    Return the geometry of the dataset as a shapely mapping structure (dictionary).
    Using fiona for IO, and only returning the first feature. Ideally the dissolved
    vector geometry should be a single feature.
    """
    with fiona.Env(
        session=AWSSession(
            aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
    ):
        with fiona.open(uri) as src:
            data = src[0]

    return data["geometry"]


def crs_info(
    uri: str, access_key: Union[str, None] = None, secret_key: Union[str, None] = None
) -> Dict[str, str]:
    """
    Retrieve the horizontal and vertical datum information.
    """
    ctx = tiledb_context(access_key, secret_key)

    ds = tiledb.open(uri, ctx=ctx)
    metadata = json.loads(ds.meta["crs_info"])

    return metadata


def move_data(
    uid,
    array_uri,
    coverage_vector_uri,
    cells_vector_uri,
    outdir_uri,
    access_key,
    secret_key,
):
    """
    Placeholder for moving data. Nothing fancy.
    """
    ctx = tiledb_context(access_key, secret_key)
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, use_listings_cache=False)

    # coverage data file
    pth = Path(urllib.parse.urlparse(coverage_vector_uri).path)
    new_coverage_uri = outdir_uri + f"{uid}_{pth.name}"
    fs.move(coverage_vector_uri, new_coverage_uri)

    # cell data file
    pth = Path(urllib.parse.urlparse(cells_vector_uri).path)
    new_cell_uri = outdir_uri + f"{uid}_{pth.name}"
    fs.move(coverage_vector_uri, new_cell_uri)

    # tiledb array (and consolidate fragments)
    new_array_uri = outdir_uri + f"{uid}_bathymetry.tiledb"
    tiledb.move(array_uri, new_array_uri, ctx)
    tiledb.consolidate(new_array_uri, ctx=ctx)

    return new_array_uri, new_coverage_uri, new_cell_uri


def prepare(
    sonar_metadata: Dict[str, Any],
    stats_metadata: Dict[str, Any],
    asb_spreadsheet_metatadata: Dict[str, Any],
    array_uri: str,
    coverage_vector_uri: str,
    cells_vector_uri: str,
    access_key: str,
    secret_key: str,
    start_end_datetimes: List[datetime],
    outdir_uri: str,
) -> Dict[str, Any]:
    """
    Prepare a STAC item metadata document.
    """
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, use_listings_cache=False)

    properties = {f"sonar:{key}": value for key, value in sonar_metadata.items()}

    # horizontal and vertical datums
    crs_dict = crs_info(array_uri, access_key, secret_key)
    for key, value in crs_dict.items():
        properties[f"sonar:{key}"] = value

    # the STAC processing extension is not currently available in pystac
    lineage = asb_spreadsheet_metatadata["survey_general"]["lineage"]
    lineage = lineage + "\nConverted GSF's to TileDB sparse array"
    properties["processing:lineage"] = lineage

    created = datetime.now(timezone.utc)

    bounding_box = [
        stats_metadata["X"]["miniumum"],
        stats_metadata["Y"]["miniumum"],
        stats_metadata["Z"]["miniumum"],
        stats_metadata["X"]["maximum"],
        stats_metadata["Y"]["maximum"],
        stats_metadata["Z"]["maximum"],
    ]

    # STAC.PC requires a list of dicts for the stats
    # eg [{stats}, {stats}]
    # a dict would be better eg {"attribute_name": {stats},}
    # STAC.PC also requires the attribute name to be inserted into the stats dict
    stats_md = stats_metadata.copy()
    for key in stats_md:
        stats_md[key]["name"] = key
    stats = [value for _, value in stats_md.items()]

    schema = create_pdal_schema(array_uri, access_key, secret_key)

    geometry = data_geometry(cells_vector_uri, access_key, secret_key)

    # in regards to the uid, assiging within here is better than passing one through
    # for the project the uid will form part of the name (quick, easy, consistent)
    # as the naming is so disparate between the providers
    # we could rename the data files, and re-register the tiledb array as
    # it is better to have it registered to make of of UDFArray funcs
    uid = uuid.uuid4()
    item = pystac.Item(
        id=str(uid),
        datetime=start_end_datetimes[1],
        geometry=geometry,
        bbox=bounding_box,
        properties=properties,
    )

    # common metadata
    item.common_metadata.title = asb_spreadsheet_metatadata["survey_general"][
        "survey_title"
    ]
    item.common_metadata.description = asb_spreadsheet_metatadata["survey_general"][
        "abstract"
    ]
    item.common_metadata.start_datetime = start_end_datetimes[0]
    item.common_metadata.end_datetime = start_end_datetimes[1]
    item.common_metadata.created = created
    item.common_metadata.instruments = [
        asb_spreadsheet_metatadata["bathy_technical"]["sensor_type"],
    ]
    item.common_metadata.providers = [
        pystac.Provider(
            name=asb_spreadsheet_metatadata["survey_citation"]["data_owner"],
            roles=[pystac.ProviderRole.PRODUCER],
        ),
        pystac.Provider(
            name=asb_spreadsheet_metatadata["survey_citation"]["custodian"],
            roles=[pystac.ProviderRole.HOST],
        ),
    ]

    # STAC extensions
    proj_ext = ProjectionExtension.ext(item, add_if_missing=True)
    pc_ext = PointcloudExtension.ext(item, add_if_missing=True)
    sci_ext = ScientificExtension.ext(item, add_if_missing=True)

    proj_ext.apply(epsg=int(crs_dict["horizontal_datum"][5:]))  # "epsg:..."

    pc_ext.apply(
        count=stats_md[list(stats_md.keys())[0]]["count"],
        type=PhenomenologyType.SONAR,
        encoding="TileDB",
        schemas=[Schema(properties=sch) for sch in schema],
        statistics=[Statistic(properties=stat) for stat in stats],
    )

    sci_ext.apply(
        citation=asb_spreadsheet_metatadata["survey_citation"]["attribution_citation"],
    )

    # moving data to new locations
    new_array_uri, new_coverage_uri, new_cells_uri = move_data(
        uid,
        array_uri,
        coverage_vector_uri,
        cells_vector_uri,
        outdir_uri,
        access_key,
        secret_key,
    )
    array_name = Path(new_array_uri).stem

    stac_md_uri = outdir_uri + f"{uid}_stac-metadata.geojson"

    item.add_asset("bathymetry_data", pystac.Asset(href=new_array_uri, roles=["data"]))
    item.add_asset(
        "data_footprint", pystac.Asset(href=new_coverage_uri, roles=["data"])
    )
    item.add_asset("sounding_density", pystac.Asset(href=new_cells_uri, roles=["data"]))
    item.add_link(
        pystac.Link(rel="self", media_type=pystac.MediaType.JSON, target=stac_md_uri)
    )

    stac_metadata_dict = item.to_dict()

    with fs.open(stac_md_uri, "w") as src:
        json.dump(stac_metadata_dict, src, indent=4, cls=Encoder)

    tiledb.cloud.register_array(
        uri=new_array_uri,
        namespace="AusSeabed", # Optional, you may register it under your username, or one of your organizations
        array_name=array_name,
        description=asb_spreadsheet_metatadata["survey_general"]["abstract"],  # Optional 
        access_credentials_name="AusSeabedGMRT-PL019"
    )

    return stac_metadata_dict
