import numpy
import tiledb
import tiledb.cloud


def cell_frequency(array_uri, cell_frequency_uri, access_key, skey):
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


token = ""

tiledb.cloud.login(token=token)

tiledb.cloud.udf.register_generic_udf(cell_frequency, name="cell_frequency", namespace="AusSeabed")
