"""Module for parameter sensitivity"""

import random
import numpy as np
import pandas as pd
from SALib.sample import saltelli, morris as morris_sampler, fast_sampler
from SALib.analyze import sobol, morris as morris_analyzer, fast
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging


class WOFOSTSensitivityAnalyzer:
    """
    A robust Sensitivity Analysis (SA) tool for WOFOST simulations.

    This class integrates SALib with the WOFOST simulation runner to perform
    global and local sensitivity analyses. It supports parallel execution,
    spatial aggregation (to filter out weather noise), and multiple SA methods.

    Attributes:
        runner: An instance of the simulation runner (Batch or Single) capable of
                spawning engine instances.
        engines (dict): A dictionary mapping location IDs to WOFOST engine instances.
    """

    def __init__(self, runner):
        """
        Initializes the analyzer and pre-loads simulation engines.

        Args:
            runner: A WOFOST simulation runner object. Must implement
                    `get_batch_rerunners()` or `get_rerunner()`.
        """
        # DISABLE PCSE LOGGING
        pcse_logger = logging.getLogger("pcse")
        pcse_logger.handlers = []
        pcse_logger.setLevel(logging.CRITICAL)

        self.runner = runner
        self.engines = {}

        print("[SA] Loading simulation engines...")
        if hasattr(runner, "get_batch_rerunners"):
            self.engines = runner.get_batch_rerunners()
        else:
            self.engines = {0: runner.get_rerunner()}

    def run_analysis(
        self,
        problem_dict,
        method="sobol",
        n_samples=128,
        n_workers=4,
        target_variable="TWSO",
        mode="global",
        sample_locations=10,
        num_levels=4,
    ):
        """
        Executes the Sensitivity Analysis workflow.

        This method generates parameter samples using SALib, runs simulations across
        selected locations in parallel, aggregates the results based on the specified
        mode, and computes sensitivity indices.

        Args:
            problem_dict (dict): The standard SALib problem definition dictionary.
                Example:
                {
                    'num_vars': 2,
                    'names': ['TSUM1', 'SPAN'],
                    'bounds': [[800, 1200], [28, 35]]
                }

            method (str, optional): The SA method to use. Options:
                - 'sobol': Variance-based (Best for interactions, computationally expensive).
                - 'morris': Elementary effects (Good for screening many parameters).
                - 'fast': Fourier Amplitude Sensitivity Test.
                Defaults to 'sobol'.

            n_samples (int, optional): The base sample size (N).
                **Note:** The actual number of simulations depends on the method:
                - Sobol: N * (2D + 2)
                - Morris: N * (D + 1)
                - FAST: N * D
                Where D is the number of parameters. Defaults to 128.

            n_workers (int, optional): Number of parallel processes to use. Defaults to 4.

            target_variable (str, optional): The output column from the simulation results
                to analyze (e.g., 'TWSO' for Total Dry Weight Storage Organs).
                Defaults to 'TWSO'.

            mode (str, optional): The spatial aggregation strategy.
                - 'global': Runs simulations on `sample_locations` for each parameter set,
                  averages the results to remove weather noise, and returns a single
                  sensitivity report for the entire region.
                - 'local': Performs a full sensitivity analysis for EACH location
                  individually. Returns a DataFrame containing indices for every location.
                Defaults to 'global'.

            sample_locations (int, optional): The number of random locations to select
                from the batch runner to represent the region. If None, uses all available
                locations (warning: this can be very slow). Defaults to 10.

            num_levels (int, optional): Number of grid levels (specific to 'morris' method).
                Defaults to 4.

        Returns:
            pd.DataFrame: A DataFrame containing the sensitivity indices.
                - For 'sobol': Columns [Parameter, ST, S1, point_id]
                - For 'morris': Columns [Parameter, mu_star, sigma, point_id]
                - For 'fast': Columns [Parameter, ST, S1, point_id]

        Raises:
            ValueError: If an unknown method is specified.
        """

        # 1. GENERATE SAMPLES (Method Specific)
        if method == "sobol":
            param_values = saltelli.sample(
                problem_dict, n_samples, calc_second_order=False
            )
        elif method == "morris":
            param_values = morris_sampler.sample(
                problem_dict, n_samples, num_levels=num_levels
            )
        elif method == "fast":
            param_values = fast_sampler.sample(problem_dict, n_samples)
        else:
            raise ValueError(
                f"Unknown method: {method}. Available methods are: {'sobol', 'morris', 'fast'}."
            )

        total_runs_per_loc = len(param_values)

        # 2. SELECT LOCATIONS
        all_locs = list(self.engines.keys())
        if sample_locations is None or sample_locations >= len(all_locs):
            selected_locs = all_locs
        else:
            selected_locs = random.sample(all_locs, sample_locations)

        print(
            f"[SA] Mode: {mode.upper()} | Params: {total_runs_per_loc} | Locations: {len(selected_locs)}"
        )
        print(f"[SA] Total Simulations: {total_runs_per_loc * len(selected_locs)}")

        # 2. PREPARE TASKS
        tasks = []
        for run_idx, row in enumerate(param_values):
            overrides = self._row_to_overrides(row, problem_dict)
            for loc_id in selected_locs:
                engine = self.engines[loc_id]
                # Payload: ( (run_idx, loc_id), engine, overrides )
                tasks.append(((run_idx, loc_id), engine, overrides, target_variable))

        # 4. EXECUTE SIMULATIONS
        results_map = {idx: {} for idx in range(total_runs_per_loc)}
        failure_count = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_sa_worker_wrapper, task) for task in tasks]
            iterator = as_completed(futures)

            for future in tqdm(iterator, total=len(futures), desc="[SA] Simulating"):
                try:
                    r_idx, l_id, val, err_msg = future.result()

                    results_map[r_idx][l_id] = val

                    if err_msg:
                        failure_count += 1
                        logging.debug(f"Failure: {err_msg}")

                except Exception as e:
                    print(f"Critical Worker Error: {e}")
                    failure_count += 1

        if failure_count > 0:
            print(f"[SA] Warning: {failure_count} simulations failed.")

        # 5. AGGREGATE & ANALYZE
        final_dfs = []

        if mode == "global":
            # Mean across locations for each parameter set
            y_aggregated = np.zeros(total_runs_per_loc)
            for idx in range(total_runs_per_loc):
                vals = list(results_map[idx].values())
                y_aggregated[idx] = np.mean(vals) if vals else 0.0

            print(f"[SA] Analyzing global (Mean) sensitivity...")
            df = self._analyze_results(problem_dict, param_values, y_aggregated, method)
            df["point_id"] = "global"
            return df

        elif mode == "local":
            print(f"[SA] Analyzing local sensitivity...")

            for loc_id in selected_locs:
                # Extract vector for this specific location
                y_local = np.zeros(total_runs_per_loc)
                for idx in range(total_runs_per_loc):
                    y_local[idx] = results_map[idx].get(loc_id, 0.0)

                df_loc = self._analyze_results(
                    problem_dict, param_values, y_local, method
                )
                df_loc["point_id"] = loc_id
                final_dfs.append(df_loc)

            return pd.concat(final_dfs, ignore_index=True)

    def _analyze_results(self, problem, X, Y, method):
        """
        Internal helper to run the specific SALib analysis function.

        Args:
            problem (dict): SALib problem dictionary.
            X (np.ndarray): The matrix of parameter samples used.
            Y (np.ndarray): The vector of simulation results.
            method (str): Analysis method ('sobol', 'morris', 'fast').

        Returns:
            pd.DataFrame: A standardized dataframe of sensitivity indices.
        """
        if method == "sobol":
            Si = sobol.analyze(problem, Y, calc_second_order=False)
            return pd.DataFrame(
                {"Parameter": problem["names"], "ST": Si["ST"], "S1": Si["S1"]}
            ).sort_values("ST", ascending=False)

        elif method == "morris":
            Si = morris_analyzer.analyze(problem, X, Y, conf_level=0.95)
            return pd.DataFrame(
                {
                    "Parameter": problem["names"],
                    "mu_star": Si["mu_star"],
                    "sigma": Si["sigma"],
                }
            ).sort_values("mu_star", ascending=False)

        elif method == "fast":
            Si = fast.analyze(problem, Y)
            return pd.DataFrame(
                {"Parameter": problem["names"], "ST": Si["ST"], "S1": Si["S1"]}
            ).sort_values("ST", ascending=False)

    def _row_to_overrides(self, row, problem):
        """
        Maps a flat row of parameter values to the WOFOST override dictionary structure.
        """
        crop_params = {}
        for i, name in enumerate(problem["names"]):
            crop_params[name] = row[i]
        return {"crop_params": crop_params, "soil_params": {}, "site_params": {}}


# WORKER WRAPPER FOR SENSITIVITY ANALYSIS
def _sa_worker_wrapper(payload):
    """
    Unpacks payload, runs engine, returns scalar target.

    Returns a Tuple of 4 items:
    (run_idx, loc_id, value, error_message)
    """
    (run_idx, loc_id), engine, overrides, target_var = payload
    try:
        # Run Simulation
        df = engine(
            crop_overrides=overrides.get("crop_params"),
            soil_overrides=overrides.get("soil_params"),
            site_overrides=overrides.get("site_params"),
        )

        if df is not None and not df.empty:
            if target_var in df.columns:
                # 1. Filter for maturity (or max DVS)
                max_dvs = df["DVS"].max()
                df = df[df["DVS"] == max_dvs]

                # 2. Extract value
                val = df[target_var].mean()  # Use mean in case of duplicate days

                # SUCCESS: Return Value + None for error
                return (run_idx, loc_id, val, None)
            else:
                return (run_idx, loc_id, np.nan, f"Column '{target_var}' missing")
        else:
            return (run_idx, loc_id, np.nan, "Empty DataFrame returned")

    except Exception as e:
        # FAILURE: Return NaN + The Error Message
        return (run_idx, loc_id, np.nan, str(e))
