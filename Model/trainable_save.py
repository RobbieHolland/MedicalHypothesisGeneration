import pytorch_lightning as pl
import glob
import torch

class TrainableSave(pl.LightningModule):
    def __init__(self):
        super().__init__()

    # def find_checkpoint(self, checkpoint_code=None):
    #     checkpoint_code = self.config.model.checkpoint_path if checkpoint_code is None else checkpoint_code
    #     glob_string = f'{self.config.pretrained_model_dir}/{checkpoint_code[0]}/*{checkpoint_code[1]}*'
    #     checkpoints = glob.glob(glob_string)
    #     if len(checkpoints) == 0:
    #         raise Exception(f'No model checkpoints in {glob_string}')
    #     return checkpoints[0]

    # def load_from_checkpoint_file(self, checkpoint_path=None):
    #     checkpoint_path = self.find_checkpoint() if checkpoint_path is None else checkpoint_path
    #     print(f'Loading pretrained model from {checkpoint_path}')
    #     checkpoint = torch.load(checkpoint_path)

    #     # Get the names of all parameters that require gradients
    #     grad_param_keys = {name for name, param in self.named_parameters() if param.requires_grad}

    #     # Check that each grad parameter has a corresponding key in the loaded state_dict
    #     loaded_keys = set(checkpoint['state_dict'].keys())
    #     missing_keys = grad_param_keys - loaded_keys
    #     if missing_keys:
    #         print(f"Warning: Missing keys in loaded state_dict: {missing_keys}")
    #     extra_keys = loaded_keys - grad_param_keys
    #     if extra_keys:
    #         print(f"Warning: Extra keys in loaded state_dict: {extra_keys}")
        
    #     self.load_state_dict(checkpoint['state_dict'], strict=False)

    def on_save_checkpoint(self, checkpoint: dict):
        grad_params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        checkpoint['state_dict'] = {k: v for k, v in self.state_dict().items() if k in grad_params}
