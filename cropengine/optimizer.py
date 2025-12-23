"""Module for parameter optimization"""

import os
import optuna
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Callable, Dict, List, Union, Optional


# GLOBAL WORKER FUNCTION
def _global_worker_task(payload):
    """
    Independent worker function for parallel execution.

    Args:
        payload (tuple): Contains (loc_id, engine_instance, overrides_dict)

    Returns:
        pd.DataFrame or None: The simulation result with an added 'point_id' column.
    """

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
    except Exception:
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

            sampler (optuna.samplers.BaseSampler): Custom Optuna sampler (e.g., TPESampler, NSGAII).

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

        # 2. PRE-LOADING PHASE
        print("[OPT] Loading simulation engines...")
        if not self.engines:
            if self.is_batch:
                self.engines = self.runner.get_batch_rerunners()
            else:
                self.engines = {0: self.runner.get_rerunner()}

        print(f"[OPT] Ready. Optimized execution for {len(self.engines)} locations.")

        # 2. DEFINE OBJECTIVE
        def objective(trial):
            # A. Get Parameters from Optuna
            overrides = search_space(trial)

            # B. Prepare Tasks for Parallel Workers
            tasks = [
                (loc_id, engine, overrides) for loc_id, engine in self.engines.items()
            ]

            results = []

            # C. Execute in Parallel (ProcessPool for True Parallelism)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                chunk_size = max(1, len(tasks) // (n_workers * 4))

                for res in executor.map(
                    _global_worker_task, tasks, chunksize=chunk_size
                ):
                    if res is not None:
                        results.append(res)

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

        # 3. CREATE STUDY
        if directions is None:
            directions = ["minimize"]

        study = optuna.create_study(directions=directions, sampler=sampler)

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
