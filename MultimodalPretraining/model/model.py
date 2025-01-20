from MultimodalPretraining.model.merlin_wrapper import MerlinWrapper

MODEL_MAP = {
    "merlin": MerlinWrapper,
}

def load_model(config):
    return MerlinWrapper(config)