from setuptools import setup

setup(
    name="bathy_datasets",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    url="https://github.com/ausseabed/bathy_datasets",
    author="AusSeabed",
    description="Prototype module for creating bathymetry datasets from the AusSeabed team",  # noqa:E501
    keywords=[
        "bathymetry",
        "metadata",
        "stac",
        "dggs",
    ],
    # packages=find_packages(),
    packages=["bathy_datasets"],
    install_requires=[
        "numpy",
        "attrs",
        "pandas",
        "numba",
        "rhealpixdggs",
        "h3",
        "structlog",
    ],
    license="Apache",
    zip_safe=False,
)
