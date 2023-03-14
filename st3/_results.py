import pandas as pd


class STResults:
    def __init__(self, results: dict, sources: list):
        """Container for results from SourceTracker."""
        self.results = results
        self.sources = sources + ["Unknown"]
        self.sinks = list(results.keys())

    def __getitem__(self, sink_id: str):
        return self.results[sink_id]

    def __len__(self):
        return len(self.results)

    def to_dataframe(self) -> pd.DataFrame:
        """Get estimated mixing proportions as Pandas DataFrame."""
        results = [
            r.variational_params_pd.filter(like="mix_prop")
            for sink, r in self.results.items()
        ]
        results = pd.concat(results)
        results.columns = self.sources
        results.index = self.sinks
        return results


class STResultsLOOCollapsed(STResults):
    def __init__(self, results: dict):
        """Container for results from SourceTrackerLOOCollapsed."""
        self.results = results
        self.sources = list(results.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """Get estimated mixing proportions as Pandas DataFrame."""
        results = []
        # For each result, get remaining sources by subsetting source list
        for i, (src, res) in enumerate(self.results.items()):
            mix_props = res.variational_params_pd.filter(like="mix_prop")
            _sources = self.sources[:i] + self.sources[i+1:] + ["Unknown"]
            mix_props.columns = _sources
            results.append(mix_props)

        results = pd.concat(results)[self.sources + ["Unknown"]]
        results.index = self.sources
        return results
