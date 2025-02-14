import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, config, models, inference_map):
        super().__init__()
        self.config = config
        self.models = models
        self.inference_map = inference_map

    def forward(self, inputs):
        intermediate_outputs = {}
        
        for key, mapping in self.inference_map.items():
            keys = key if isinstance(key, tuple) else (key,)  # Ensure keys is always iterable

            collected_inputs = []
            for k in keys:
                if k in intermediate_outputs:
                    collected_inputs.append(intermediate_outputs[k])
                elif k in inputs:
                    collected_inputs.append(inputs[k])
                else:
                    raise KeyError(f"Missing key '{k}' in inputs or intermediate_outputs")

            output = mapping['forward_model'](collected_inputs)
            intermediate_outputs[mapping['output_field']] = output

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

    multimodal_model = MultimodalModel(config, inference_map)

if __name__ == "__main__":
    main()