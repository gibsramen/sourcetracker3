import biom
import numpy as np
import pandas as pd


def collapse_data(
    table: biom.Table,
    metadata: pd.DataFrame,
    sourcesink_column: str = "SourceSink",
    env_column: str = "Env",
    source_name: str = "source",
    sink_name: str = "sink"
) -> (pd.DataFrame, pd.DataFrame):
    tbl_data = table.to_dataframe(dense=True).T

    source_data = []
    md_grouped = (
        metadata[metadata[sourcesink_column] == source_name]
        .groupby(env_column)
    )
    for env, env_md in md_grouped:
        env_samples = env_md.index
        env_data = tbl_data.loc[env_samples].sum(axis=0)
        env_data = pd.Series(env_data, name=env)
        source_data.append(env_data)

    source_data = pd.concat(source_data, axis=1).T.astype(int)
    source_data = biom.Table(
        source_data.T.values,
        observation_ids=source_data.columns,
        sample_ids=source_data.index
    )

    sinks = metadata[metadata[sourcesink_column] == sink_name].index
    sink_tbl_data = tbl_data.loc[sinks]
    sink_tbl = biom.Table(
        sink_tbl_data.T.values.astype(int),
        observation_ids=sink_tbl_data.columns,
        sample_ids=sink_tbl_data.index
    )
    return source_data, sink_tbl
