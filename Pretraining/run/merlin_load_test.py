import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device available")

import sys
# sys.path.append('/dataNAS/people/lblankem/contrastive-3d')
sys.path.append('/dataNAS/people/akkumar/contrastive-3d')

from contrastive_3d.models import load_pretrained_model

dataset_config = {
    # "dataset": "stanford",
    "dataset": "stanford_disease_prediction_all",
    "fraction_train_data": 1.0,
    "per_device_train_batch_size": 2,
    "per_device_val_batch_size": 2,
    "per_device_test_batch_size": 2,
}

from contrastive_3d.datasets import dataloaders
dataloaders = dataloaders.get_dataloaders(dataset_config)
train_loader, val_loader, test_loader = dataloaders

import pandas as pd
for dl in dataloaders:
    df = pd.DataFrame([{**{'image': d['image']}, **{f'label_{i+1}': v for i, v in enumerate(d['label'])}} for d in dl.dataset.data])

batch = next(iter(train_loader))
# Image (3D), path to segmentations, labels (1692 list of batch size lists, likely onehot), findings (structured text per region), findings_list (list of 14 regions of batch size findings for that region)
print(batch)

model = load_pretrained_model.build_imagenet_stage1_stage2_mtl_seg_clip_100(n_classes=1692)
print(model)

# Returns two vectors of size batch size x N, where N is 1692 for classification and 512 for the classification head - what about the original 2048?
features = model(batch['image'])
print(features)
