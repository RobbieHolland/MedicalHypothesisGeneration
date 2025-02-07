import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from sklearn.utils.class_weight import compute_sample_weight
import gc 
from imblearn.under_sampling import RandomUnderSampler
import threading
from concurrent.futures import ThreadPoolExecutor

class LogisticRegressionAnalysis:
    def __init__(self, config, sae_output, phecode_columns, active_features, output_dir):
        self.config = config
        self.sae_output = sae_output
        self.phecode_columns = phecode_columns
        self.active_features = active_features
        self.output_dir = output_dir

        self.X = self.sae_output[self.active_features].values
        self.X_mean, self.X_var = self.X.mean(0), self.X.var(0)
        self.X_normalized = (self.X - self.X_mean) / np.sqrt(self.X_var + 1e-8)

        self.Y = self.sae_output[self.phecode_columns].values
        x = 3

    def compute_coefficients_and_pvalues(self):
        # Extract feature matrix (X) and label matrix (Y) from sae_output

        # Add intercept (constant term) once
        X = sm.add_constant(self.X_normalized)

        # Preallocate results for efficiency
        n_labels = self.Y.shape[1]
        n_features = X.shape[1] - 1  # Exclude the intercept for the output
        coef_matrix = np.zeros((n_labels, n_features))
        pvals_matrix = np.zeros((n_labels, n_features))

        # Fit logistic regression for each label (one-vs-rest)
        for i in tqdm(range(n_labels)):
            model = sm.Logit(self.Y[:, i], X)
            results = model.fit_regularized(alpha=0.05, L1_wt=0)
            
            # Store coefficients and p-values, excluding intercept
            coef_matrix[i, :] = results.params[1:]
            pvals_matrix[i, :] = results.pvalues[1:]

        return coef_matrix, pvals_matrix

    def perform_analysis(self):
        coef_matrix, pvals_matrix = self.fit_glm()
        # coef_matrix, pvals_matrix = self.compute_coefficients_and_pvalues()

        # Build DataFrame objects
        coef_df = pd.DataFrame(
            coef_matrix, 
            index=self.phecode_columns, 
            columns=self.active_features
        )
        pvals_df = pd.DataFrame(
            pvals_matrix, 
            index=self.phecode_columns, 
            columns=self.active_features
        )

        return coef_df, pvals_df

    from concurrent.futures import ThreadPoolExecutor

    def fit_glm(self, k=100):
        coef_matrix = np.full((self.Y.shape[1], self.X.shape[1]), np.nan)
        pvals_matrix = np.full((self.Y.shape[1], self.X.shape[1]), np.nan)

        def process_feature(i, j):
            rus = RandomUnderSampler(
                sampling_strategy=lambda y: {
                    cls: min(cnt, np.bincount(y).min() * 4) if cnt == max(np.bincount(y)) 
                    else np.bincount(y).min()
                    for cls, cnt in zip(*np.unique(y, return_counts=True))
                },
                random_state=self.config.seed
            )
            
            X_j = self.X[:, j].reshape(-1, 1)
            Y_i = self.Y[:, i]

            valid_mask = ~np.isnan(Y_i) & ~np.isnan(X_j).flatten()
            X_j = X_j[valid_mask]
            Y_i = Y_i[valid_mask].astype(np.int64)
            
            try:
                X_resampled, Y_resampled = rus.fit_resample(X_j, Y_i)
            except ValueError:
                print(f"Skipping X_{j}, Y_{i} due to resampling issues.")
                return

            if (X_resampled != 0).sum() < k:
                return

            X_resampled = sm.add_constant(X_resampled)

            results = None
            def fit_model():
                nonlocal results
                model = sm.GLM(Y_resampled, X_resampled, family=sm.families.Binomial())
                results = model.fit(tol=1e-6, disp=False, maxiter=10)
            
            fit_thread = threading.Thread(target=fit_model)
            fit_thread.start()
            fit_thread.join(timeout=10)

            if fit_thread.is_alive():
                print(f"Timeout reached for X_{j}, Y_{i}. Skipping.")
                return

            if results is not None:
                coef_matrix[i, j] = results.params[1]
                pvals_matrix[i, j] = results.pvalues[1]
            else:
                print(f"Skipping coefficients for X_{j}, Y_{i} due to timeout or failure.")

        for i in tqdm(range(self.Y.shape[1])):
            with ThreadPoolExecutor() as executor:
                executor.map(lambda j: process_feature(i, j), range(self.X.shape[1]))

        return coef_matrix, pvals_matrix

    def firth_logistic(self, X, y):
        model = sm.Logit(y, X)
        res = model.fit_regularized(method='l1', alpha=1e-5)  # Small penalty to improve convergence
        return res.params, res.pvalues

# # Usage
# coef_matrix, pvals_matrix = fit_glm_vectorized(X, Y)
