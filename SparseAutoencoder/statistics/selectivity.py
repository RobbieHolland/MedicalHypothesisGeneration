import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SelectivityAnalysis:
    def __init__(self, config, sae_output, phecode_columns, active_features, output_dir):
        self.sae_output = sae_output
        self.phecode_columns = phecode_columns
        self.active_features = active_features
        self.output_dir = output_dir
        self.config = config

    def compute_selectivity_matrix(self, data, phecode_columns, active_features):
        # Placeholder for the actual computation logic
        pass

    def plot_selectivity_matrix(self, matrix):
        plt.imshow(matrix, aspect='auto', cmap='viridis')
        plt.colorbar()

    def save_plot(self, matrix, title, filename):
        self.plot_selectivity_matrix(matrix)
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def perform_analysis(self):
        # 1) Compute original selectivity
        avg_sel_original, max_sel_original = self.compute_selectivity_matrix(
            self.sae_output, self.phecode_columns, self.active_features
        )

        # Process & plot original average selectivity
        avg_sel_mat = pd.DataFrame(avg_sel_original, index=self.phecode_columns, columns=self.active_features)
        self.save_plot(avg_sel_mat, "Average Selectivity Matrix", 'average_selectivity_matrix.jpg')

        # Process & plot original max selectivity
        max_sel_mat = pd.DataFrame(max_sel_original, index=self.phecode_columns, columns=self.active_features)
        self.save_plot(max_sel_mat, "Maximum Selectivity Matrix", 'maximum_selectivity_matrix.jpg')

        # 2) Bootstrap null distributions
        bootstrap_avg = []
        bootstrap_max = []
        for _ in tqdm(range(self.config.task.n_bootstrap)):
            boot_data = self.sae_output.sample(frac=1, replace=True)
            avg_sel_boot, max_sel_boot = self.compute_selectivity_matrix(
                boot_data, self.phecode_columns, self.active_features
            )
            bootstrap_avg.append(avg_sel_boot)
            bootstrap_max.append(max_sel_boot)

        bootstrap_avg = np.stack(bootstrap_avg, axis=0)  # shape: (n_bootstrap, n_phecodes, n_features)
        bootstrap_max = np.stack(bootstrap_max, axis=0)

        # 3) Compute p-values (fraction of bootstrap samples >= observed)
        avg_p_values = np.mean(bootstrap_avg >= avg_sel_original[None, ...], axis=0)
        max_p_values = np.mean(bootstrap_max >= max_sel_original[None, ...], axis=0)

        # 4) Merge p-values with original selectivities and save
        avg_sel_df = pd.DataFrame(avg_sel_original, index=self.phecode_columns, columns=self.active_features)
        max_sel_df = pd.DataFrame(max_sel_original, index=self.phecode_columns, columns=self.active_features)
        avg_pvals_df = pd.DataFrame(avg_p_values, index=self.phecode_columns, columns=self.active_features)
        max_pvals_df = pd.DataFrame(max_p_values, index=self.phecode_columns, columns=self.active_features)

        # Save the results to output directory
        avg_sel_df.to_csv(os.path.join(self.output_dir, 'average_selectivity.csv'))
        max_sel_df.to_csv(os.path.join(self.output_dir, 'maximum_selectivity.csv'))
        avg_pvals_df.to_csv(os.path.join(self.output_dir, 'average_pvalues.csv'))
        max_pvals_df.to_csv(os.path.join(self.output_dir, 'maximum_pvalues.csv'))

        return avg_sel_df, avg_pvals_df