from abc import ABC, abstractmethod
from pkg_resources import resource_filename
from functools import partial
from multiprocessing import Pool
from typing import Callable

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

        :param source_metadata: Metadata containing information about sources
        :type source_metadata: pd.DataFrame
        """
        self.features = list(source_table.ids("observation"))
        self.sources = list(source_table.ids("sample"))
        self.num_features = source_table.shape[0]
        self.source_data = source_table.matrix_data.toarray().T
        self.unknown_mu_prior = unknown_mu_prior
        self.unknown_kappa_prior = unknown_kappa_prior

    def fit_variational(self, sinks: biom.Table, jobs: int = 1, **kwargs):
        inf_function = partial(MODEL.variational, **kwargs)
        return self._fit_multiple(sinks, jobs=jobs, inf_function=inf_function,
                                  inf_type="variational")

    def fit_mcmc(self, sinks: biom.Table, jobs: int = 1, chains: int = 4,
                 iter_warmup: int = 500, iter_sampling: int = 500, **kwargs):
        inf_function = partial(
            MODEL.sample,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            **kwargs
        )
        return self._fit_multiple(sinks, jobs=jobs, inf_function=inf_function,
                                  inf_type="mcmc")

    def _fit_multiple(self, sinks: biom.Table, inf_function: Callable,
                      inf_type: str, jobs=1):
        # Make sure order of features is the same
        sink_data = (
            sinks
            .filter(self.features, "observation", inplace=False)
            .matrix_data
            .toarray()
            .T
        )
        with Pool(jobs) as p:
            results = p.map(
                partial(self._fit_single, inf_function=inf_function),
                sink_data
            )

        if inf_type == "mcmc":
            results = STResultsMCMC(results, self.sources, sinks.ids())
        if inf_type == "variational":
            results = STResultsVariational(results, self.sources, sinks.ids())

        return results

    def _fit_single(self, sink: np.array, inf_function: Callable):
        data = {
            "N": self.num_features,
            "x": sink.astype(int),
            "K": self.num_sources,
            "y": self.source_data.astype(int),
            "unknown_mu": self.unknown_mu_prior,
            "unknown_kappa": self.unknown_kappa_prior
        }
        results = inf_function(data=data)
        return results


class STResults(ABC):
    def __init__(self, results: list, sources: list, sinks: list):
        self.results = results
        self.sources = sources + ["Unknown"]
        self.sinks = sinks

    def __len__(self):
        return len(self.results)

    @abstractmethod
    def to_dataframe(self):
        pass


class STResultsVariational(STResults):
    def __init__(self, results: list, sources: list, sinks: list):
        super().__init__(results, sources, sinks)

    def to_dataframe(self):
        results = [
            x.variational_params_pd.filter(like="mix_prop")
            for x in self.results
        ]
        results = pd.concat(results)
        results.columns = self.sources
        results.index = self.sinks
        return results


class STResultsMCMC(STResults):
    def __init__(self, results: list, sources: list, sinks: list):
        super().__init__(results, sources, sinks)

    def to_dataframe(self):
        results = [
            (
                x.draws_pd()
                .filter(like="mix_prop")
                .assign(sink=sink)
            )
            for x, sink in zip(self.results, self.sinks)
        ]
        results = pd.concat(results).reset_index(names=["draw"])

        num_sources = len(self.sources)
        source_map = {
            f"mix_prop[{i+1}]": source
            for i, source in enumerate(self.sources)
        }
        source_map["mix_prop[{num_sources+1}]"] = "Unknown"
        results = results.rename(columns=source_map)
        return results
