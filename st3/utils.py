import biom
import pandas as pd


def collapse_data(
    table: biom.Table,
    metadata: pd.DataFrame,
    sourcesink_column: str = "SourceSink",
    env_column: str = "Env",
    source_name: str = "source",
    sink_name: str = "sink"
) -> (biom.Table, biom.Table):
    """Collapse feature table into source and sink tables.

    Example feature table:

        #OTU ID A       B       C       D       X       Y       Z
        Taxa_1  4.0     4.0     5.0     1.0     7.0     1.0     4.0
        Taxa_2  2.0     2.0     5.0     4.0     1.0     6.0     2.0
        Taxa_3  5.0     0.0     2.0     2.0     3.0     4.0     0.0

    Example metadata:

              Env SourceSink
         A  Env_1     source
         B  Env_1     source
         C  Env_2     source
         D  Env_2     source
         X   sink       sink
         Y   sink       sink
         Z   sink       sink

    Returned source table:

        #OTU ID Env_1   Env_2
        Taxa_1  8.0     6.0
        Taxa_2  4.0     9.0
        Taxa_3  5.0     4.0

    Returned sink table:

        #OTU ID X       Y       Z
        Taxa_1  7.0     1.0     4.0
        Taxa_2  1.0     6.0     2.0
        Taxa_3  3.0     4.0     0.0

    :param table: Feature table containing both trainking and cink samples
    :type table: biom.Table

    :param metadata: Sample metadata
    :type metadata: pd.DataFrame

    :param sourcesink_column: Name of column in sample metadata denoting which
        samples are sources and which are sinks, default 'SourceSink'
    :type sourcesink_column: str

    :param env_column: Name of column in sample metadata with environment
        names, default 'Env'
    :type env_column: str

    :param source_name: Level in sourcesink_column corresponding to source
        samples, default 'source'
    :type source_name: str

    :param sink_name: Level in sourcesink_column corresponding to sink
        samples, default 'sink'
    :type sink_name: str

    :returns: Source table, sink table
    :rtype: tuple
    """
    sinks = metadata[metadata[sourcesink_column] == sink_name].index
    training = metadata[metadata[sourcesink_column] == source_name].index

    sink_tbl = table.filter(sinks, inplace=False)
    source_tbl = table.filter(training, inplace=False)

    source_map = metadata[env_column].to_dict()
    source_tbl = source_tbl.collapse(lambda s, m: source_map[s], norm=False)

    return source_tbl, sink_tbl
