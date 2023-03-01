from cmdstanpy import CmdStanVB
import numpy as np
import pytest

from st3.model import SourceTracker
from st3.utils import collapse_data


@pytest.fixture
def st3_model(example_data):
    table, metadata = example_data
    source_tbl, sink_tbl = collapse_data(table, metadata)
    return sink_tbl, SourceTracker(source_tbl)


@pytest.fixture
def st3_model_results(st3_model):
    sink_tbl, model = st3_model
    return model.fit(sink_tbl)


def test_results_type(st3_model_results):
    assert all([isinstance(r, CmdStanVB) for r in st3_model_results])


def test_results_to_df(st3_model, st3_model_results):
    sink_tbl, _ = st3_model
    results_df = st3_model_results.to_dataframe()
    assert results_df.shape == (10, 6)

    np.testing.assert_allclose(results_df.sum(1), np.ones(10), atol=0.001)

    sink_ids = sink_tbl.ids("sample")
    assert (sink_ids == results_df.index).all()

    exp_srcs = [f"SRC_{x+1}" for x in range(5)] + ["Unknown"]
    assert all(results_df.columns == exp_srcs)
