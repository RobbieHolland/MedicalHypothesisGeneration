import hydra
import os
import pandas as pd
import wandb
from util.plotting import savefig
import yaml

class Figures:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(config.base_dir, 'Analysis/output', f'sweep_id={config.task.wandb_sweep_id}')
        os.makedirs(self.output_dir, exist_ok=True)

        # Authenticate W&B
        wandb.login(key=config.wandb_api_token)

        # Initialize API
        api = wandb.Api()

        # Fetch sweep runs
        sweep = api.sweep(f"{config.wandb_entity}/MedicalHypothesisGeneration-Analysis_run/{config.task.wandb_sweep_id}")
        runs = sweep.runs

        # Extract relevant data
        data = []
        for run in runs:
            summary = run.summary._json_dict  # Final metrics
            data.append({**run.config, **summary})

        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Assuming df has columns: 'task.outputs', 'data', 'config.outcome'
        df['task.outputs'] = df['task.outputs'].apply(lambda l: l[0])
        df = df.loc[~df['validation_max_auc_epoch'].isna()]
        best_runs = df.loc[df.groupby(['task.outputs', 'data'])["validation_max_auc_epoch"].idxmax()]
        # best_runs = df.loc[df['task.lr'] == 0.0005]

        with open("Analysis/slurm/sweep/linear_evaluation.yaml", "r") as file:
            sweep_config = yaml.safe_load(file)
        output_order = [o[0] for o in sweep_config['parameters']['task.outputs']['values']]

        heatmap_data = best_runs.pivot(index='task.outputs', columns='data', values=config.task.outcome)
        heatmap_data = heatmap_data.reindex(output_order, columns=config.task.data_order)

        # Plot heatmap
        plt.figure(figsize=(10, 1 * len(heatmap_data)))
        sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5)

        # Labels
        plt.xlabel("Data")
        plt.ylabel("Task Outputs")
        plt.title("Performance Heatmap (AUC)")

        # Show plot
        save_path = os.path.join(self.output_dir, 'linear_evaluation_datatype_prognosis.jpg')
        savefig(save_path)

    def analyze(self):
        pass

@hydra.main(config_path="../../config", config_name="default", version_base=None)
def main(config):
    analysis = Figures(config)
    analysis.analyze()

if __name__ == "__main__":
    main()
