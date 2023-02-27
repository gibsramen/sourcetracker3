import pathlib
from pkg_resources import resource_filename

import biom
import pandas as pd
import pytest

TEST_DATA_DIR = pathlib.Path(resource_filename("st3", "tests/test_data"))
TEST_TBL = TEST_DATA_DIR / "table.biom"
TEST_MD = TEST_DATA_DIR / "metadata.tsv"


@pytest.fixture(scope="session")
def example_data():
    table = biom.load_table(TEST_TBL)
    metadata = pd.read_table(TEST_MD, sep="\t", index_col=0)
    return table, metadata
