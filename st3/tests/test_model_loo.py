import numpy as np
import pandas as pd
import pytest

from st3.model import SourceTrackerLOO, SourceTrackerLOOCollapsed


@pytest.fixture
def example_data_loo(example_data):
    table, metadata = example_data
    samps_to_keep = metadata[metadata["SourceSink"] == "source"].index
    table = table.filter(samps_to_keep, inplace=False)
    metadata = metadata.loc[list(table.ids())]
    return table, metadata


def test_st3_loo(example_data_loo):
    table, metadata = example_data_loo
    st3_loo = SourceTrackerLOO(table, metadata)
    results = st3_loo.fit()
    results_df = results.to_dataframe()

    exp_cols = [f"SRC_{x+1}" for x in range(5)] + ["Unknown"]
    assert list(results_df.columns) == exp_cols

    assert (metadata.index == results_df.index).all()


def test_st3_loo_collapsed(example_data_loo):
    table, metadata = example_data_loo
    st3_loo_coll = SourceTrackerLOOCollapsed(table, metadata)
    results = st3_loo_coll.fit()
    results_df = results.to_dataframe()

    exp_cols = [f"SRC_{x+1}" for x in range(5)] + ["Unknown"]
    assert list(results_df.columns) == exp_cols

    exp_sources = [f"SRC_{x+1}" for x in range(5)]
    assert list(results_df.index) == exp_sources

    for src, row in results_df.iterrows():
        assert pd.isna(row[src])
        np.testing.assert_almost_equal(row.sum(), 1, decimal=4)
