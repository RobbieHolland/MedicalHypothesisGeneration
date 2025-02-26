import wandb 
import os

def linear_eval_path(config, project=None, group=None, sweep_id='no_sweep'):
    sweep_id = wandb.run.sweep_id if wandb.run and wandb.run.sweep_id else sweep_id
    project = project if project else wandb.run.project
    group = group if group else wandb.run.group

    experiment_path = os.path.join(config.data.type, "-".join(config.data.inputs), config.data.model.name, config.task.outputs[0])

    save_path = os.path.join(config.pretrained_model_dir, project, group, sweep_id, experiment_path)
    return save_path