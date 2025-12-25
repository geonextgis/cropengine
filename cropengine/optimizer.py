"""Module for parameter optimization"""

import os
import optuna
from optuna.trial import FixedTrial
import pandas as pd
import logging
from typing import Callable, Dict, List, Union, Optional
from joblib import Parallel, delayed


# GLOBAL WORKER FUNCTION
def _global_worker_task(payload):
    """
    Independent worker function for parallel execution.

    Args:
        payload (tuple): Contains (loc_id, engine_instance, overrides_dict)

    Returns:
        pd.DataFrame or None: The simulation result with an added 'point_id' column.
    """
    # DISABLE PCSE LOGGING
    pcse_logger = logging.getLogger("pcse")
    pcse_logger.handlers = []
    pcse_logger.setLevel(logging.CRITICAL)

    loc_id, engine, overrides = payload
    try:
        df = engine(
            crop_overrides=overrides.get("crop_params"),
            soil_overrides=overrides.get("soil_params"),
            site_overrides=overrides.get("site_params"),
        )
        if df is not None and not df.empty:
            df["point_id"] = loc_id
            return df
    except Exception as e:
        print(f"Worker Error loc {loc_id}: {e}")
        return None
    return None


class WOFOSTOptimizer:
    """
    A generalized optimizer for WOFOST simulations using Optuna.

    Features:
    - **Parallel Execution**: Uses ProcessPoolExecutor to bypass the GIL and utilize all CPU cores.
    - **Memory Efficient**: Loads simulation engines (Weather/Soil/Agro) into RAM once and reuses them.
    - **Multi-Objective**: Supports optimizing multiple targets simultaneously (Pareto optimization).
    - **Agnostic**: Works with both Single-Location Runners and Batch Runners.
    """

    def __init__(self, runner, observed_data):
        """
        Instantiate WOFOSTOptimizer.

        Args:
        runner: An instance of WOFOSTCropSimulationRunner or WOFOSTCropSimulationBatchRunner.
        observed_data (pd.DataFrame): Ground truth data used by the loss function.
        """
        self.runner = runner
        self.observed_data = observed_data
        self.is_batch = hasattr(runner, "get_batch_rerunners")
        self.engines = {}

    def _get_sampler(
        self, sampler_input: Union[str, optuna.samplers.BaseSampler, None]
    ) -> optuna.samplers.BaseSampler:
        """
        Helper to resolve the sampler from a string name or object.
        """
        if sampler_input is None:
            return None  # Let Optuna choose default (usually TPESampler)

        if isinstance(sampler_input, optuna.samplers.BaseSampler):
            return sampler_input

        if isinstance(sampler_input, str):
            name = sampler_input.lower().strip()

            try:
                if name == "random":
                    return optuna.samplers.RandomSampler()

                elif name == "tpe":
                    return optuna.samplers.TPESampler()

                elif name == "cmaes":
                    # Requires 'cma' package
                    return optuna.samplers.CmaEsSampler()

                elif name == "nsgaii":
                    return optuna.samplers.NSGAIISampler()

                elif name == "nsgaiii":
                    return optuna.samplers.NSGAIIISampler()

                elif name == "qmc":
                    # Quasi-Monte Carlo (requires Scipy)
                    return optuna.samplers.QMCSampler()

                elif name == "bruteforce":
                    return optuna.samplers.BruteForceSampler()

                elif name == "grid":
                    return optuna.samplers.GridSampler(
                        search_space={}
                    )  # Note: GridSampler requires search space passed later or usually managed by study

                elif name == "botorch":
                    # Requires 'botorch' package
                    from optuna.integration import BoTorchSampler

                    return BoTorchSampler()

                elif name == "gp":
                    # Gaussian Process Sampler (Requires 'botorch' & 'scipy')
                    return optuna.samplers.GPSampler()

                else:
                    raise ValueError(f"Unknown sampler name: '{sampler_input}'.")

            except ImportError as e:
                # Catch missing dependency errors (e.g., missing botorch or cma)
                raise ImportError(
                    f"Could not initialize sampler '{name}'. Missing dependency: {e}. Please install the required package (e.g., 'pip install botorch' or 'pip install cma')."
                )

        raise TypeError(
            "Sampler must be a string name or an optuna.samplers.BaseSampler object."
        )

    def get_best_params(self, study: optuna.Study, search_space: Callable) -> Dict:
        """
        Retrieves the optimized parameters from the study, reconstructing any
        complex structures (lists/tables) defined in the search space.

        Args:
            study (optuna.Study): The completed optimization study.
            search_space (callable): The original search space function used for optimization.
                                     Required to reconstruct complex parameters (lists/tables)
                                     from the scalar values stored in the study.

        Returns:
            dict: A dictionary of parameter overrides (e.g., {'crop_params': {...}})
                  containing the best values found during optimization.
        """
        best_overrides = search_space(FixedTrial(study.best_params))

        return best_overrides

    def optimize(
        self,
        search_space: Callable[[optuna.Trial], Dict],
        loss_func: Callable[[pd.DataFrame, pd.DataFrame], Union[float, List[float]]],
        n_trials: int = 100,
        n_workers: int = 4,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        directions: Optional[List[str]] = None,
        output_folder: Optional[str] = None,
    ) -> optuna.Study:
        """
        Runs the optimization loop.

        Args:
            search_space (callable): A function that takes an Optuna `trial` object
                                     and returns a dictionary of parameter overrides.
                                     Example structure:
                                     {'crop_params': {'TSUM1': 1000}, 'soil_params': {...}}

            loss_func (callable): A function that takes (df_simulated, df_observed).
                                  Returns a float (single-objective) or list of floats (multi-objective).

            n_trials (int): Number of optimization trials to run.

            n_workers (int): Number of parallel processes to spawn.

            sampler (str | optuna.samplers.BaseSampler | None):
                The optimization strategy. Supported strings:

                **Standard:**
                - "TPE": Tree-structured Parzen Estimator (Default, good general purpose).
                - "Random": Pure random search.

                **Advanced (May require extra packages):**
                - "GP": Gaussian Process Sampler. Excellent for expensive simulations. (Requires `botorch`).
                - "CmaEs": Covariance Matrix Adaptation. Good for continuous global optima. (Requires `cma`).
                - "BoTorch": Bayesian Optimization. (Requires `botorch`).

                **Multi-Objective:**
                - "NSGAII": Standard for Pareto optimization.
                - "NSGAIII": For many-objective problems (3+ targets).

                **Grid/Deterministic:**
                - "BruteForce": Tries ALL combinations.
                - "Grid": Tries specified grid points.
                - "QMC": Quasi-Monte Carlo.

            directions (list[str]): Optimization directions.
                                    Default is ["minimize"].
                                    For multi-objective, use e.g., ["minimize", "maximize"].

            output_folder (str, optional): Path to a folder where simulation results
                                           for EACH trial will be saved (e.g., 'trial_0.csv').
                                           If None, results are not saved to disk.

        Returns:
            optuna.Study: The completed study object containing best params and trials.
        """
        # 1. SETUP OUTPUT FOLDER
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            print(f"[OPT] Saving all trial outputs to: {output_folder}")

        # 2. RESOLVE SAMPLER
        optuna_sampler = self._get_sampler(sampler)
        if optuna_sampler:
            print(f"[OPT] Using Sampler: {optuna_sampler.__class__.__name__}")

        # 3. PRE-LOADING PHASE
        print("[OPT] Loading simulation engines...")
        if not self.engines:
            if self.is_batch:
                self.engines = self.runner.get_batch_rerunners()
            else:
                self.engines = {0: self.runner.get_rerunner()}

        print(f"[OPT] Ready. Optimized execution for {len(self.engines)} locations.")

        # 4. DEFINE OBJECTIVE
        def objective(trial):
            # A. Get Parameters from Optuna
            overrides = search_space(trial)

            # B. Prepare Tasks for Parallel Workers
            tasks = [
                (loc_id, engine, overrides) for loc_id, engine in self.engines.items()
            ]

            results = []

            # C. Execute in Parallel using JOBLIB
            try:
                # Parallel returns a list of results in order
                results_raw = Parallel(n_jobs=n_workers, backend="loky")(
                    delayed(_global_worker_task)(task) for task in tasks
                )

                # Filter out None values (failed runs)
                results = [res for res in results_raw if res is not None]

            except Exception as e:
                logging.error(f"[OPT] Parallel Execution Error: {e}")
                results = []

            # D. Validation
            if not results:
                if directions and len(directions) > 1:
                    return [float("inf")] * len(directions)
                return float("inf")

            # E. Aggregation & Loss Calculation
            try:
                # 1. Merge all location results into one DataFrame
                df_sim_all = pd.concat(results, ignore_index=True)

                if output_folder:
                    file_path = os.path.join(output_folder, f"trial_{trial.number}.csv")
                    df_sim_all.to_csv(file_path, index=False)

                # 2. Compute Loss (User Function)
                loss = loss_func(df_sim_all, self.observed_data)
                return loss

            except Exception as e:
                logging.error(f"[OPT] Loss Calculation Error: {e}")
                if directions and len(directions) > 1:
                    return [float("inf")] * len(directions)
                return float("inf")

        # 5. CREATE STUDY
        if directions is None:
            directions = ["minimize"]

        study = optuna.create_study(directions=directions, sampler=optuna_sampler)

        print(
            f"[OPT] Starting {len(directions)}-objective optimization with {n_trials} trials..."
        )
        study.optimize(objective, n_trials=n_trials)

        print("[OPT] Optimization Finished.")

        if len(directions) == 1:
            print("Best params:", study.best_params)
        else:
            print(
                f"Pareto front found with {len(study.best_trials)} optimal solutions."
            )

        return study
