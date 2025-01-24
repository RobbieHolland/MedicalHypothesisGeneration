import wandb
import os

def init_wandb(config):
    os.environ["WANDB_API_KEY"] = config.wandb_api_token

    wandb.init(
        project=config.wandb_project,
        group=config.task.wandb_group,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        config=dict(config)
    )
