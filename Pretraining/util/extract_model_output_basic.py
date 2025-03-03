import torch
import numpy as np
import gc
from tqdm import tqdm  # Import tqdm for progress bar

def extract_vectors_for_split(config, dataloader, model, input_field, fields, save_path="./embeddings"):
    model.eval()  # Ensure model is in evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_fields = {field: [] for field in fields}
    all_outputs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            inputs = batch[0][input_field].to(device)  # Move input to GPU/CPU
            batch_output = model.latent(inputs).cpu().numpy()  # Convert to NumPy
            
            all_outputs.append(batch_output)  # Store batch outputs
            
            for field in fields:
                all_fields[field].extend(batch[1][field])  # Collect metadata fields
            
    return {
        "output": np.concatenate(all_outputs, axis=0),
        **{key: np.array(all_fields[key]) for key in all_fields}
    }
