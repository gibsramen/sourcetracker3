import pytest

import st3.utils as utils


class TestCollapseData:
    @pytest.fixture(autouse=True)
    def setup_example_data(self, example_data):
        self.table, self.metadata = example_data
        self.source_tbl, self.sink_tbl = (
            utils.collapse_data(self.table, self.metadata)
        )

    def test_collapse_data(self):
        # Should be 20 taxa, 5 (known) sources, 10 sinks
        assert self.source_tbl.shape == (20, 5)
        assert self.sink_tbl.shape == (20, 10)

        source_taxa = self.source_tbl.ids("observation")
        sink_taxa = self.sink_tbl.ids("observation")
        assert (source_taxa == sink_taxa).all()

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
