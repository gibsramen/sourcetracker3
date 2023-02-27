from functools import partial
from multiprocessing import Pool
import pathlib
from pkg_resources import resource_filename
from tempfile import TemporaryDirectory

import biom
from cmdstanpy import CmdStanModel, CmdStanVB
from joblib import Parallel, delayed
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

        :param unknown_mu_prior: Prior belief for unknown proportion, default
            0.2
        :type unknown_mu_prior: float

        :param unknown_kappa_prior: Prior belief for kappa parameter for beta
            proprotion distribution, default 10
        :type unknown_kappa_prior: float
        """
        self.features = list(source_table.ids("observation"))
        self.sources = list(source_table.ids("sample"))
        self.num_features, self.num_sources = source_table.shape
        self.source_data = source_table.matrix_data.toarray().T
        self.unknown_mu_prior = unknown_mu_prior
        self.unknown_kappa_prior = unknown_kappa_prior

    def fit(self, sinks: biom.Table, jobs: int = 1, parallel_args: dict = None,
            temp_dir: pathlib.Path = None, **kwargs) -> "STResults":
        """Fit SourceTracker model on multiple sink samples.

        :param sinks: Table of sink samples
        :type sinks: biom.Table

        :param jobs: Number of jobs to run in parallel, default 1
        :type jobs: int

        :param parallel_args: Arguments to pass to joblib.Parallel
        :type parallel_args: dict

        :param temp_dir: Temporary directory in which to save intermediate
            CSVs creating during sampling
        :type temp_dir: pathlib.Path

        :param **kwargs: Keyword arguments to pass to CmdStanModel.variational

        :returns: Results of each sink's fitted model
        :rtype: st3.model.STResults
        """
        # Make sure order of features is the same
        sink_data = (
            sinks
            .filter(self.features, "observation", inplace=False)
            .matrix_data
            .toarray()
            .T
        )

        func = partial(self._fit_single, temp_dir=temp_dir, **kwargs)
        parallel_args = parallel_args or dict()
        results = Parallel(n_jobs=jobs, **parallel_args)(
            delayed(func)(vals) for vals in sink_data
        )

        results = STResults(results, self.sources, sinks.ids())
        return results

    def _fit_single(self, sink: np.array, temp_dir: pathlib.Path,
                    **kwargs) -> CmdStanVB:
        """Fit a single sink sample.

        :param sink: Array of taxa counts in same order as source table
        :type sink: np.array

        :param temp_dir: Temporary directory in which to save intermediate
            CSVs creating during sampling
        :type temp_dir: pathlib.Path

        :param **kwargs: Keyword arguments to pass to CmdStanModel.variational

        :returns: Model fitted through variational inference
        :rtype: CmdStanPy.CmdStanVB
        """
        data = {
            "N": self.num_features,
            "x": sink.astype(int),
            "K": self.num_sources,
            "y": self.source_data.astype(int),
            "unknown_mu": self.unknown_mu_prior,
            "unknown_kappa": self.unknown_kappa_prior
        }

        if temp_dir is not None:
            with TemporaryDirectory(dir=temp_dir) as output_dir:
                results = MODEL.variational(data=data, output_dir=output_dir,
                                            **kwargs)
        else:
            results = MODEL.variational(data=data, **kwargs)
        return results


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
