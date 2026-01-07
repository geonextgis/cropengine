"""Module for data assimilation"""

import copy

import numpy as np
import pandas as pd
from pcse.base import ParameterProvider
from pcse.input import ExcelWeatherDataProvider, YAMLAgroManagementReader
from tqdm import tqdm

from cropengine.models import get_model_class


class WOFOSTEnKF:
    """
    Implements data assimilation with an Ensemble Kalman Filter (EnKF) for WOFOST.
    """

    def __init__(self, runner, ensemble_size=50, seed=None):
        """
        Args:
            runner: An initialized WOFOSTCropSimulationRunner instance.
                    The runner must have 'prepare_system' called already.
            ensemble_size (int): Number of ensemble members.
            seed (int): Random seed for reproducibility.
        """
        self.runner = runner
        self.ensemble_size = ensemble_size
        self.rng = np.random.default_rng(seed)
        self.ensemble = []

        # Load base configuration from the runner's workspace
        self.base_config = self._load_base_config()

    def _load_base_config(self):
        """Loads weather, agro, and parameters from the runner's workspace."""

        if not hasattr(self.runner, "files"):
            raise RuntimeError(
                "Runner workspace not prepared. Run 'prepare_system' first."
            )

        weather = ExcelWeatherDataProvider(
            self.runner.files["weather"], force_reload=True
        )
        agro = YAMLAgroManagementReader(self.runner.files["agro"])

        soil = self.runner._load_param_dict("soil_params", "value")
        site = self.runner._load_param_dict("site_params", "value")
        crop = self.runner._load_param_dict("crop_params", "value")

        params = ParameterProvider(soildata=soil, sitedata=site, cropdata=crop)

        return {"weather": weather, "agro": agro, "params": params}

    def setup_ensemble(self, param_std):
        """
        Initializes the ensemble.

        Args:
            param_std (dict): dict to perturb specific parameters
                              e.g. {'TSUM1': 50, 'SPAN': 2}.
        """
        ModelClass = get_model_class(self.runner.model_name)
        self.ensemble = []

        for i in range(self.ensemble_size):
            # 1. Perturb Parameters
            config = copy.deepcopy(self.base_config)

            for p_name, p_std in param_std.items():
                # Perturb crop parameters
                curr_val = config["params"][p_name]
                new_val = self.rng.normal(curr_val, p_std)
                config["params"][p_name] = max(0, new_val)

            # 2. Initialize Model Member
            member = ModelClass(config["params"], config["weather"], config["agro"])
            self.ensemble.append(member)

    def run_assimilation(
        self, observations_df, observation_std=0.5, state_vars=["LAI", "SM"]
    ):
        """
        Runs the Multistate EnKF loop.

        Args:
            observations_df (pd.DataFrame): Must contain 'date' and columns matching state_vars.
            observation_std (dict): Uncertainty for each variable. {'LAI': 0.1, 'SM': 0.05}.
            state_vars (list): The full state vector to track and update.
        """

        # 1. Prepare Observations
        obs_df = observations_df.copy()
        obs_df["date"] = pd.to_datetime(obs_df["date"]).dt.date
        obs_df = obs_df.sort_values("date")

        # Get unique dates to iterate over
        obs_dates = sorted(obs_df["date"].unique())

        pbar = tqdm(total=len(obs_dates) + 1, desc="Running Data Assimilation")

        active_members = list(self.ensemble)

        # === LOOP 1: JUMP BETWEEN OBSERVATIONS ===
        for obs_date in obs_dates:

            if not active_members:
                break

            # A. Advance all ensemble members to this date
            for member in active_members[:]:
                current_date = member.day
                days_to_go = (obs_date - current_date).days

                if days_to_go > 0:
                    member.run(days=days_to_go)

                    if member.flag_terminate:
                        active_members.remove(member)
                        continue

            # === ASSIMILATION STEP ===
            survivors = [m for m in active_members if m.day == obs_date]

            if survivors:
                # Get the row for today
                daily_row = obs_df[obs_df["date"] == obs_date].iloc[0]

                # 1. Identify what is observed today
                # We scan the row for any variable in state_vars that is not NaN
                current_observations = {}
                for var in state_vars:
                    if var in daily_row and pd.notnull(daily_row[var]):
                        current_observations[var] = daily_row[var]

                # Only update if we actually found relevant data
                if current_observations:
                    self._apply_filter(
                        survivors, current_observations, observation_std, state_vars
                    )

            pbar.update(1)

        # === LOOP 2: FINISH SIMULATION ===
        # Run remaining days until maturity/harvest for survivors
        for member in active_members:
            member.run_till_terminate()

        pbar.update(1)
        pbar.close()
        return self._collect_results()

    def _apply_filter(self, members, observations, obs_std_dict, state_vars):
        """
        The Kalman Filter Update.
        Updates model states based on the difference between predicted and observed.
        """
        # A. Collect Predicted States
        n_members = len(members)
        n_states = len(state_vars)
        n_obs = len(observations)

        # 1. Build State Matrix (A) [n_states x n_members]
        A = np.zeros((n_states, n_members))
        for i, member in enumerate(members):
            for j, var in enumerate(state_vars):
                val = member.get_variable(var)
                A[j, i] = val if val is not None else 0.0

        # 2. Build Measurement Matrix (H) [n_obs x n_states]
        H = np.zeros((n_obs, n_states))
        obs_keys = list(observations.keys())  # e.g. ['LAI', 'SM']

        for row_idx, obs_var in enumerate(obs_keys):
            try:
                col_idx = state_vars.index(obs_var)
                H[row_idx, col_idx] = 1.0
            except ValueError:
                pass

        # 3. Build Perturbed Observation Matrix (D) [n_obs x n_members]
        D = np.zeros((n_obs, n_members))
        R_diag = []  # Diagonal elements for Observation Error Covariance

        for row_idx, obs_var in enumerate(obs_keys):
            value = observations[obs_var]
            std = obs_std_dict.get(obs_var, 0.1)  # Default error if missing
            R_diag.append(std**2)

            # Perturb the single observation to create an ensemble of observations
            D[row_idx, :] = self.rng.normal(value, std, n_members)

        R = np.diag(R_diag)  # [n_obs x n_obs]

        # 4. Calculate EnKF Statistics
        # Ensemble Mean and Anomalies
        A_mean = np.mean(A, axis=1, keepdims=True)
        A_prime = A - A_mean

        # Predicted Observation Anomalies (Y = HA')
        Y = H @ A_prime

        # Innovation Covariance (S = Y Y.T / (N-1) + R)
        S = (1.0 / (n_members - 1)) * (Y @ Y.T) + R

        # Cross Covariance (PHt = A' Y.T / (N-1))
        PHt = (1.0 / (n_members - 1)) * (A_prime @ Y.T)

        # Kalman Gain (K = PHt * S^-1)
        # Use pseudo-inverse for stability if S is singular
        try:
            K = PHt @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = PHt @ np.linalg.pinv(S)

        # 5. Update States (A_a = A + K(D - HA))
        HA = H @ A
        Innovations = D - HA
        A_analysis = A + (K @ Innovations)

        # 6. Inject Back into Model
        for i, member in enumerate(members):
            for j, var in enumerate(state_vars):
                new_val = A_analysis[j, i]
                new_val = max(0.0, new_val)
                member.set_variable(var, new_val)

    def _collect_results(self):
        """Aggregates results from all members."""
        dfs = []
        for i, m in enumerate(self.ensemble):
            df = pd.DataFrame(m.get_output())
            df["member_id"] = i
            dfs.append(df)

        full_df = pd.concat(dfs)

        return full_df
