from abc import ABC, abstractmethod
from functools import partial
import pathlib
from pkg_resources import resource_filename
from tempfile import TemporaryDirectory

import biom
from cmdstanpy import CmdStanModel, CmdStanVB
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from ._results import STResults, STResultsLOOCollapsed
from .utils import collapse_data

MODEL_PATH = resource_filename("st3", "stan/sourcetracker.stan")
MODEL = CmdStanModel(stan_file=MODEL_PATH)


class STBase(ABC):
    def __init__(
        self,
        num_features: int,
        num_sources: int,
        unknown_mu_prior: float = 0.2,
        unknown_kappa_prior: float = 10
    ):
        """Base SourceTracker instance

        :param num_features: Number of features in dataset
        :type num_features: int

        :param num_sources: Number of sources to consider
        :type num_sources: int

        :param unknown_mu_prior: Prior belief for unknown proportion, default
            0.2
        :type unknown_mu_prior: float

        :param unknown_kappa_prior: Prior belief for kappa parameter for beta
            proprotion distribution, default 10
        :type unknown_kappa_prior: float
        """
        self.num_features = num_features
        self.num_sources = num_sources
        self.unknown_mu_prior = unknown_mu_prior
        self.unknown_kappa_prior = unknown_kappa_prior

    def _fit_single(
        self,
        sink: np.array,
        source_data: np.array,
        temp_dir: pathlib.Path = None,
        **kwargs
    ) -> CmdStanVB:
        """Fit a single sink sample.

        :param sink: Array of taxa counts in same order as source table
        :type sink: np.array

        :param source_data: Array of taxa counts by sources
        :type source_data: np.array

        :param temp_dir: Temporary directory in which to save intermediate
            CSVs creating during sampling
        :type temp_dir: pathlib.Path

        :param **kwargs: Keyword arguments to pass to CmdStanModel.variational

        :returns: Model fitted through variational inference
        :rtype: cmdstanpy.CmdStanVB
        """
        data = {
            "N": self.num_features,
            "x": sink.astype(int),
            "K": self.num_sources,
            "y": source_data.astype(int),
            "unknown_mu": self.unknown_mu_prior,
            "unknown_kappa": self.unknown_kappa_prior
        }

        if temp_dir is not None:
            # Create a subdirectory in temp_dir for each sink sample
            with TemporaryDirectory(dir=temp_dir) as output_dir:
                results = MODEL.variational(data=data, output_dir=output_dir,
                                            **kwargs)
        else:
            results = MODEL.variational(data=data, **kwargs)
        return results

    @abstractmethod
    def fit(self):
        """Abstract method for fitting sink data"""
        pass


class SourceTracker(STBase):
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
        super().__init__(
            num_features=source_table.shape[0],
            num_sources=source_table.shape[1],
            unknown_mu_prior=unknown_mu_prior,
            unknown_kappa_prior=unknown_kappa_prior
        )
        self.features = list(source_table.ids("observation"))
        self.sources = list(source_table.ids("sample"))
        self.source_data = source_table.matrix_data.T.toarray()

    def fit(
        self,
        sinks: biom.Table,
        jobs: int = 1,
        parallel_args: dict = None,
        temp_dir: pathlib.Path = None,
        **kwargs
    ) -> STResults:
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
        :rtype: st3._results.STResults
        """
        func = partial(self._fit_single, source_data=self.source_data,
                       temp_dir=temp_dir, **kwargs)
        parallel_args = parallel_args or dict()

        # Make sure order of features is the same
        sink_data = sinks.filter(self.features, "observation", inplace=False)

        results = Parallel(n_jobs=jobs, **parallel_args)(
            delayed(func)(vals) for vals in sink_data.iter_data()
        )
        results = dict(zip(sinks.ids(), results))

        results = STResults(results, self.sources)
        return results


class SourceTrackerLOO(STBase):
    def __init__(
        self,
        table: biom.Table,
        metadata: pd.DataFrame,
        sourcesink_column: str = "SourceSink",
        env_column: str = "Env",
        source_name: str = "source",
        sink_name: str = "sink",
        unknown_mu_prior: float = 0.2,
        unknown_kappa_prior: float = 10
    ):
        """Initialize SourceTracker instance for leave-out-sample out

        Creates a SourceTracker instance for leaving each training sample out.

        :param table: Feature table of training samples by features
        :type table: biom.Table

        :param metadata: Sample metadata
        :type metadata: pd.DataFrame

        :param sourcesink_column: Name of column in sample metadata denoting
            which samples are sources and which are sinks, default 'SourceSink'
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

        :param unknown_mu_prior: Prior belief for unknown proportion, default
            0.2
        :type unknown_mu_prior: float

        :param unknown_kappa_prior: Prior belief for kappa parameter for beta
            proprotion distribution, default 10
        :type unknown_kappa_prior: float
        """
        self.metadata = metadata

        self.sources = list(
            metadata[metadata[sourcesink_column] == source_name][env_column]
            .unique()
        )
        self.table = table
        self.samples = table.ids()
        self.source_map = self.metadata[env_column].to_dict()

        super().__init__(
            num_features=table.shape[0],
            num_sources=len(self.sources),
            unknown_mu_prior=unknown_mu_prior,
            unknown_kappa_prior=unknown_kappa_prior
        )

    def fit(
        self,
        jobs: int = 1,
        parallel_args: dict = None,
        temp_dir: pathlib.Path = None,
        **kwargs
    ) -> STResults:
        """Fit SourceTracker model on each training sample as hold-out.

        :param jobs: Number of jobs to run in parallel, default 1
        :type jobs: int

        :param parallel_args: Arguments to pass to joblib.Parallel
        :type parallel_args: dict

        :param temp_dir: Temporary directory in which to save intermediate
            CSVs creating during sampling
        :type temp_dir: pathlib.Path

        :param **kwargs: Keyword arguments to pass to CmdStanModel.variational

        :returns: Results of each sink's fitted model
        :rtype: st3._results.STResults
        """
        func = partial(self._fit_single, temp_dir=temp_dir, **kwargs)
        parallel_args = parallel_args or dict()

        results = Parallel(n_jobs=jobs, **parallel_args)(
            delayed(func)(samp_name) for samp_name in self.samples
        )
        results = dict(zip(self.samples, results))

        results = STResults(results, self.sources)
        return results

    def _fit_single(
        self,
        sample_name: str,
        temp_dir: pathlib.Path,
        **kwargs
    ) -> CmdStanVB:
        """Fit a single hold-out sample.

        :param sample_name: Name of training sample to use as hold-out
        :type sample_name: str

        :param temp_dir: Temporary directory in which to save intermediate
            CSVs creating during sampling
        :type temp_dir: pathlib.Path

        :param **kwargs: Keyword arguments to pass to CmdStanModel.variational

        :returns: Model fitted through variational inference
        :rtype: cmdstanpy.CmdStanVB
        """
        source_data, sink = self._get_source_sink(sample_name)
        results = super()._fit_single(
            sink, source_data, temp_dir=temp_dir, **kwargs
        )
        return results

    def _get_source_sink(self, sample_name: str) -> (np.array, np.array):
        """Gets training table and hold-out sample data

        :param sample_name: Name of training sample to use as hold-out
        :type sample_name: str

        :returns: Training source table, held-out sink data
        :rtype: (np.array, np.arrary)
        """
        sink = self.table.data(sample_name)
        non_sink_ids = [x for x in self.samples if x != sample_name]
        source_tbl = self.table.filter(non_sink_ids, inplace=False)

        source_data = (
            source_tbl.collapse(lambda s, m: self.source_map[s], norm=False)
            .matrix_data.T.toarray()
        )
        return source_data, sink


class SourceTrackerLOOCollapsed(STBase):
    def __init__(
        self,
        table: biom.Table,
        metadata: pd.DataFrame,
        sourcesink_column: str = "SourceSink",
        env_column: str = "Env",
        source_name: str = "source",
        unknown_mu_prior: float = 0.2,
        unknown_kappa_prior: float = 10
    ):
        """Initialize SourceTracker instance for leave-out-source out

        Creates a SourceTracker instance for leaving each source out.

        :param table: Feature table of training samples by features
        :type table: biom.Table

        :param metadata: Sample metadata
        :type metadata: pd.DataFrame

        :param sourcesink_column: Name of column in sample metadata denoting
            which samples are sources and which are sinks, default 'SourceSink'
        :type sourcesink_column: str

        :param env_column: Name of column in sample metadata with environment
            names, default 'Env'
        :type env_column: str

        :param source_name: Level in sourcesink_column corresponding to source
            samples, default 'source'
        :type source_name: str

        :param unknown_mu_prior: Prior belief for unknown proportion, default
            0.2
        :type unknown_mu_prior: float

        :param unknown_kappa_prior: Prior belief for kappa parameter for beta
            proprotion distribution, default 10
        :type unknown_kappa_prior: float
        """
        self.metadata = metadata

        self.sources = list(
            metadata[metadata[sourcesink_column] == source_name][env_column]
            .unique()
        )
        self.table = table
        self.samples = table.ids()
        self.source_map = self.metadata[env_column].to_dict()
        self.sourcesink_column = sourcesink_column
        self.env_column = env_column
        self.source_name = source_name

        super().__init__(
            num_features=table.shape[0],
            num_sources=len(self.sources) - 1,  # Subtract 1 for LOO
            unknown_mu_prior=unknown_mu_prior,
            unknown_kappa_prior=unknown_kappa_prior
        )

    def fit(
        self,
        jobs: int = 1,
        parallel_args: dict = None,
        temp_dir: pathlib.Path = None,
        **kwargs
    ) -> STResultsLOOCollapsed:
        """Fit SourceTracker model on each source as hold-out.

        :param jobs: Number of jobs to run in parallel, default 1
        :type jobs: int

        :param parallel_args: Arguments to pass to joblib.Parallel
        :type parallel_args: dict

        :param temp_dir: Temporary directory in which to save intermediate
            CSVs creating during sampling
        :type temp_dir: pathlib.Path

        :param **kwargs: Keyword arguments to pass to CmdStanModel.variational

        :returns: Results of each source's fitted model
        :rtype: st3._results.STResultsLOOCollapsed
        """
        func = partial(self._fit_single, temp_dir=temp_dir, **kwargs)
        parallel_args = parallel_args or dict()

        results = Parallel(n_jobs=jobs, **parallel_args)(
            delayed(func)(source) for source in self.sources
        )
        results = dict(zip(self.sources, results))

        results = STResultsLOOCollapsed(results)
        return results

    def _fit_single(
        self,
        left_out_source: str,
        temp_dir: pathlib.Path,
        **kwargs
    ) -> CmdStanVB:
        """Fit a single sink sample.

        :param left_out_source: Source to hold-out during model fitting
        :type left_out_source: str

        :param temp_dir: Temporary directory in which to save intermediate
            CSVs creating during sampling
        :type temp_dir: pathlib.Path

        :param **kwargs: Keyword arguments to pass to CmdStanModel.variational

        :returns: Model fitted through variational inference
        :rtype: cmdstanpy.CmdStanVB
        """
        source_data, sink_data = self._leave_source_out(left_out_source)
        results = super()._fit_single(
            sink_data, source_data, temp_dir=temp_dir, **kwargs
        )
        return results

    def _leave_source_out(self, left_out_source: str) -> (np.array, np.array):
        """Get training and sink data for held-out source

        :param left_out_source: Source to hold-out during model fitting
        :type left_out_source: str

        :returns: Training source table, held-out source data
        :rtype: (np.array, np.arrary)
        """
        md_copy = self.metadata.copy()
        source_map = {
            x: "source" if x != left_out_source else "sink"
            for x in self.sources
        }
        md_copy["LOO"] = md_copy[self.env_column].map(source_map)

        source_data, sink_data = collapse_data(
            self.table,
            md_copy,
            "LOO",
            self.env_column
        )
        sink_data = sink_data.sum("observation")
        source_data = source_data.matrix_data.T.toarray()
        return source_data, sink_data
