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


def test_st3_loo_collapsed(example_data_loo):
    table, metadata = example_data_loo
    st3_loo_coll = SourceTrackerLOOCollapsed(table, metadata)
    results = st3_loo_coll.fit()
    results_df = results.to_dataframe()
