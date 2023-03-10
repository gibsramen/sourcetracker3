import pandas as pd


class STResults:
    def __init__(self, results: list, sources: list, sinks: list):
        """Container for results from SourceTracker."""
        self.results = results
        self.sources = sources + ["Unknown"]
        self.sinks = sinks

    def __getitem__(self, index: int):
        return self.results[index]

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def to_dataframe(self) -> pd.DataFrame:
        """Get estimated mixing proportions as Pandas DataFrame."""
        results = [
            x.variational_params_pd.filter(like="mix_prop")
            for x in self.results
        ]
        results = pd.concat(results)
        results.columns = self.sources
        results.index = self.sinks
        return results


class STResultsLOOCollapsed(STResults):
    def __init__(self, results: list, sources: list):
        """Container for results from SourceTrackerLOOCollapsed."""
        self.results = results
        self.sources = sources

    def to_dataframe(self) -> pd.DataFrame:
        """Get estimated mixing proportions as Pandas DataFrame."""
        results = []
        # For each result, get remaining sources by subsetting source list
        for i, (src, res) in enumerate(zip(self. sources, self.results)):
            mix_props = res.variational_params_pd.filter(like="mix_prop")
            _sources = self.sources[:i] + self.sources[i+1:] + ["Unknown"]
            mix_props.columns = _sources
            results.append(mix_props)

        results = pd.concat(results)[self.sources + ["Unknown"]]
        results.index = self.sources
        return results
