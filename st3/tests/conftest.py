import os

import biom
import pandas as pd
import pytest

TEST_TBL = os.path.join(os.path.dirname(__file__), "test_data/table.biom")
TEST_MD = os.path.join(os.path.dirname(__file__), "test_data/metadata.tsv")


@pytest.fixture(scope="session")
def example_data():
    table = biom.load_table(TEST_TBL)
    metadata = pd.read_table(TEST_MD, sep="\t", index_col=0)
    return table, metadata
