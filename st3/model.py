from pkg_resources import resource_filename
from multiprocessing import Pool

import biom
from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd

MODEL_PATH = resource_filename("st3", "stan/feast.stan")
MODEL = CmdStanModel(stan_file=MODEL_PATH)


class SourceTracker:
    def __init__(
        self,
        source_table: biom.Table,
        unknown_mu_prior: float = 0.2,
        unknown_kappa_prior: float = 10
    ):
        """Initialize SourceTracker instance

        Creates a SourceTracker instance with the source data

        :param source_table: Feature table of sources by features
        :type source_table: biom.Table
        """
        self.features = list(source_table.ids("observation"))
        self.sources = list(source_table.ids("sample"))
        self.num_features, self.num_sources = source_table.shape
        self.source_data = source_table.matrix_data.toarray().T
        self.unknown_mu_prior = unknown_mu_prior
        self.unknown_kappa_prior = unknown_kappa_prior

    def fit(self, sinks: biom.Table, jobs: int = 1):
        # Make sure order of features is the same
        sink_data = (
            sinks
            .filter(self.features, "observation", inplace=False)
            .matrix_data
            .toarray()
            .T
        )
        with Pool(jobs) as p:
            results = p.map(self._fit_single, sink_data)

        results = STResults(results, self.sources, sinks.ids())
        return results

    def _fit_single(self, sink: np.array):
        data = {
            "N": self.num_features,
            "x": sink.astype(int),
            "K": self.num_sources,
            "y": self.source_data.astype(int),
            "unknown_mu": self.unknown_mu_prior,
            "unknown_kappa": self.unknown_kappa_prior
        }
        results = MODEL.variational(data=data)
        return results


class STResults:
    def __init__(self, results: list, sources: list, sinks: list):
        self.results = results
        self.sources = sources + ["Unknown"]
        self.sinks = sinks

    def __len__(self):
        return len(self.results)

    def to_dataframe(self):
        results = [
            x.variational_params_pd.filter(like="mix_prop")
            for x in self.results
        ]
        results = pd.concat(results)
        results.columns = self.sources
        results.index = self.sinks
        return results
