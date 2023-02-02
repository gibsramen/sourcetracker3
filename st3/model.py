from pkg_resources import resource_filename
from functools import partial
from multiprocessing import Pool
from typing import Callable

import arviz as az
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
        self.features = source_table.ids("observation")
        self.sources = source_table.ids("sample")
        self.num_features, self.num_sources = source_table.shape
        self.source_data = source_table.matrix_data.toarray().T
        self.unknown_mu_prior = unknown_mu_prior
        self.unknown_kappa_prior = unknown_kappa_prior

    def fit_variational(self, sinks: biom.Table, jobs: int = 1, **kwargs):
        inf_function = partial(MODEL.variational, **kwargs)
        return self._fit_multiple(sinks, jobs=jobs, inf_function=inf_function)

    def fit_mcmc(self, sinks: biom.Table, jobs: int = 1, chains: int = 4,
                 iter_warmup: int = 500, iter_sampling: int = 500, **kwargs):
        inf_function = partial(
            MODEL.sample,
            chains=chains,
            iter_warmpu=iter_warmup,
            iter_sampling=iter_sampling,
            **kwargs
        )
        return self._fit_multiple(sinks, jobs=jobs, inf_function=inf_function)

    def _fit_multiple(self, sinks: biom.Table, inf_function: Callable, jobs=1):
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
