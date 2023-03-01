import biom
import numpy as np
import pandas as pd
import pytest

import st3.utils as utils


class TestCollapseData:
    @pytest.fixture(autouse=True)
    def setup_example_data(self):
        tbl_data = np.array([
            [4.0, 4.0, 5.0, 1.0, 7.0, 1.0, 4.0],
            [2.0, 2.0, 5.0, 4.0, 1.0, 6.0, 2.0],
            [5.0, 0.0, 2.0, 2.0, 3.0, 4.0, 0.0],
        ])
        sample_ids = ["A", "B", "C", "D", "X", "Y", "Z"]
        taxa_ids = ["Taxa_1", "Taxa_2", "Taxa_3"]

        self.table = biom.Table(tbl_data, sample_ids=sample_ids,
                                observation_ids=taxa_ids)

        sourcesink = ["source"] * 4 + ["sink"] * 3
        env = ["Env_1", "Env_1", "Env_2", "Env_2"] + ["sink"] * 3
        self.metadata = pd.DataFrame(dict(Env=env, SourceSink=sourcesink),
                                     index=sample_ids)
        self.source_tbl, self.sink_tbl = (
            utils.collapse_data(self.table, self.metadata)
        )

        self.exp_source_data = np.array([
            [8.0, 6.0],
            [4.0, 9.0],
            [5.0, 4.0]
        ])

        self.exp_sink_data = np.array([
            [7.0, 1.0, 4.0],
            [1.0, 6.0, 2.0],
            [3.0, 4.0, 0.0]
        ])

    def test_collapse_data(self):
        # Should be 3 taxa, 2 (known) sources, 3 sinks
        assert self.source_tbl.shape == (3, 2)
        assert self.sink_tbl.shape == (3, 3)

        source_taxa = self.source_tbl.ids("observation")
        sink_taxa = self.sink_tbl.ids("observation")
        assert (source_taxa == sink_taxa).all()

        np.testing.assert_equal(
            self.exp_source_data, self.source_tbl.matrix_data.todense()
        )
        np.testing.assert_equal(
            self.exp_sink_data, self.sink_tbl.matrix_data.todense()
        )

    def test_sourcesink_column(self):
        metadata_new = self.metadata.copy()
        col_map = {"SourceSink": "NewName"}
        metadata_new = metadata_new.rename(columns=col_map)

        new_source_tbl, new_sink_tbl = utils.collapse_data(
            self.table,
            metadata_new,
            sourcesink_column="NewName"
        )
        assert new_source_tbl == self.source_tbl
        assert new_sink_tbl == self.sink_tbl

    def test_diff_env_column(self):
        metadata_new = self.metadata.copy()
        col_map = {"Env": "Environment"}
        metadata_new = metadata_new.rename(columns=col_map)

        new_source_tbl, new_sink_tbl = utils.collapse_data(
            self.table,
            metadata_new,
            env_column="Environment"
        )
        assert new_source_tbl == self.source_tbl
        assert new_sink_tbl == self.sink_tbl

    def test_diff_source_sink_names(self):
        metadata_new = self.metadata.copy()
        metadata_new["SourceSink"] = metadata_new["SourceSink"].map(
            {"source": "A", "sink": "B"}
        )
        new_source_tbl, new_sink_tbl = utils.collapse_data(
            self.table,
            metadata_new,
            source_name="A",
            sink_name="B"
        )
        assert new_source_tbl == self.source_tbl
        assert new_sink_tbl == self.sink_tbl
