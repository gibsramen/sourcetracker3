import biom
import pandas as pd


def collapse_data(
    table: biom.Table,
    metadata: pd.DataFrame,
    sourcesink_column: str = "SourceSink",
    env_column: str = "Env",
    source_name: str = "source",
    sink_name: str = "sink"
) -> (pd.DataFrame, pd.DataFrame):
    sinks = metadata[metadata[sourcesink_column] == sink_name].index
    training = metadata[metadata[sourcesink_column] == source_name].index

    sink_tbl = table.filter(sinks, inplace=False)
    source_tbl = table.filter(training, inplace=False)

    source_map = metadata[env_column].to_dict()
    source_tbl = source_tbl.collapse(lambda s, m: source_map[s], norm=False)

    return source_tbl, sink_tbl
