import pytest

from st3.model import SourceTrackerLOO


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
    # source, sink = st3_loo._get_source_sink("SRC_1_SAMP_1")
    results = st3_loo.fit()
    results_df = results.to_dataframe()
    print(results_df)
    assert 0
