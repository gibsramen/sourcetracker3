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
    source_tbl = []
    md_grouped = (
        metadata[metadata[sourcesink_column] == source_name]
        .groupby(env_column)
    )
    for env, env_md in md_grouped:
        env_samples = env_md.index
        env_data = table.filter(env_samples, inplace=False).sum("observation")
        env_data = pd.Series(env_data, name=env)
        source_tbl.append(env_data)

    source_tbl = pd.concat(source_tbl, axis=1).T.astype(int)
    source_tbl = biom.Table(
        source_tbl.T.values,
        observation_ids=table.ids("observation"),
        sample_ids=source_tbl.index
    )

    sinks = metadata[metadata[sourcesink_column] == sink_name].index
    sink_tbl = table.filter(sinks, inplace=False)
    return source_tbl, sink_tbl
