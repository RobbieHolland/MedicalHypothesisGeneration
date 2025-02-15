import torch
import torch.nn as nn

# Updated Multimodal Model
class MultimodalModel(nn.Module):
    def __init__(self, config, models, inference_map):
        super().__init__()
        self.config = config
        self.models = models

        # Convert inference_map into an nn.ModuleDict for proper save/load behavior
        self.inference_map = nn.ModuleDict({
            str(key): mapping['forward_model'] for key, mapping in inference_map.items()
        })

        # Store metadata separately (compress flags, output fields)
        self.inference_metadata = {
            str(key): {'output_field': mapping['output_field'], 'compress': mapping['compress']}
            for key, mapping in inference_map.items()
        }

    def update_inference_map(self, key, forward_model, output_field, compress=False):
        self.inference_map[str(key)] = forward_model
        self.inference_metadata[str(key)] = {'output_field': output_field, 'compress': compress}

    def remove_compressed_entries(self):
        for k in [k for k in self.inference_metadata if self.inference_metadata[k]['compress']]:
            del self.inference_map[k]
            del self.inference_metadata[k]

    def forward(self, inputs):
        intermediate_outputs = {}

        for key, metadata in self.inference_metadata.items():
            keys = eval(key) if key.startswith("(") else (key,)  # Convert tuple keys from string back to tuple
            
            collected_inputs = []
            for k in keys:
                if k in intermediate_outputs:
                    collected_inputs.append(intermediate_outputs[k])
                elif k in inputs:
                    collected_inputs.append(inputs[k])
                else:
                    raise KeyError(f"Missing key '{k}' in inputs or intermediate_outputs")

            # Run forward pass using registered nn.Module
            output = self.inference_map[key](collected_inputs if len(collected_inputs) > 1 else collected_inputs[0])
            intermediate_outputs[metadata['output_field']] = output

        return intermediate_outputs

import hydra
@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(config):
    from Model.get_model import get_model
    from Analysis.run.linear_evaluation import LinearEvaluation
    model = get_model(config, specific_model_name='merlin')

    linear_model_path = '/dataNAS/people/rholland/MedicalHypothesisGeneration/pretrained_models/MedicalHypothesisGeneration-Analysis_run/LinearEvaluation/mu057prc/abdominal_ct_text_embeddings/ihd/clean-sweep-4/epoch=223-step=4704.ckpt'
    linear_model = LinearEvaluation.load_from_checkpoint(linear_model_path)

    inference_map = {
        'image': {'output_field': 'merlin/image', 'compress': True, 'forward_model': lambda x: model.latent(x)},
        'findings': {'output_field': 'merlin/text', 'compress': True, 'forward_model': lambda x: model.model.encode_text(x)},
        ('merlin/image', 'merlin/text'): {'output_field': 'ckd_prediction', 'compress': False, 'forward_model': lambda x: linear_model(x)}
    }

    # multimodal_model = MultimodalModel(config, inference_map)

if __name__ == "__main__":
    main()