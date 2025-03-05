import math
import sys

import torch
from torch.nn import ReplicationPad3d
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torchvision
import copy
from Model.concept_model import ConceptModel

from contrastive_3d.models.inflated_convnets_pytorch.src import inflate
from contrastive_3d.utils import window_level

# torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)

class MerlinImageToImage(ConceptModel):
    def __init__(self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False, return_skips=True, vision_ssl=False, classifier_ssl=False, multihead=False, hidden_dim=2048):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(ConceptModel, self).__init__()
        self.return_skips = return_skips
        self.conv_class = conv_class

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2)

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)
        self.vision_ssl = vision_ssl
        self.classifier_ssl = classifier_ssl
        self.multihead = multihead
        self.hidden_dim = hidden_dim

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            if self.multihead:
                self.classifiers = nn.ModuleList()  # Store multiple classifiers
                for i in range(class_nb):  # Assuming num_classifiers is defined
                    self.classifiers.append(torch.nn.Conv3d(
                        in_channels=self.hidden_dim,
                        out_channels=1,
                        kernel_size=(1, 1, 1),
                        bias=True))
            else:
                self.classifier = torch.nn.Conv3d(
                    in_channels=self.hidden_dim,
                    out_channels=class_nb,
                    kernel_size=(1, 1, 1),
                    bias=True)            
            
            self.contrastive_head = torch.nn.Conv3d(
                in_channels=self.hidden_dim,
                out_channels=512,
                kernel_size=(1, 1, 1),
                bias=True)
        elif self.classifier_ssl:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            if self.multihead:
                self.classifiers = nn.ModuleList()
                for i in range(class_nb):
                    self.classifiers.append(torch.nn.Conv3d(
                        in_channels=self.hidden_dim,
                        out_channels=1,
                        kernel_size=(1, 1, 1),
                        bias=True))
            else: 
                self.classifier = torch.nn.Conv3d(
                    in_channels=self.hidden_dim,
                    out_channels=class_nb,
                    kernel_size=(1, 1, 1),
                    bias=True)
        elif vision_ssl:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.rotation_head = torch.nn.Conv3d(
                in_channels=self.hidden_dim,
                out_channels=4,
                kernel_size=(1, 1, 1),
                bias=True)
            self.contrastive_head = torch.nn.Conv3d(
                in_channels=self.hidden_dim,
                out_channels=512,
                kernel_size=(1, 1, 1),
                bias=True)
            
            self.conv = nn.Sequential(
                nn.Conv3d(self.hidden_dim, self.hidden_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 2, self.hidden_dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 4, self.hidden_dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 8, self.hidden_dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 16, self.hidden_dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(self.hidden_dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(self.hidden_dim // 16, 1, kernel_size=1, stride=1),
            )
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate.inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim)
            self.fc = inflate.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        skips = []
        # Note: If using nnUNet, then line 125 needs to be commented out.
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        # x = window_level.apply_window_level(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        # if x.requires_grad:
        #     x = checkpoint.checkpoint(self.layer1, x)
        #     x = checkpoint.checkpoint(self.layer2, x)
        #     x = checkpoint.checkpoint(self.layer3, x)
        #     x = checkpoint.checkpoint(self.layer4, x)
        # else:
        # x = checkpoint.checkpoint(self.layer1, x)
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer2, x)
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer3, x)
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        # x = checkpoint.checkpoint(self.layer4, x)
        # if self.return_skips:
        #     skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.layer1(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        x = self.layer2(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        x = self.layer3(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        x = self.layer4(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))


        if self.conv_class:

            x_features = self.avgpool(x)
            
            if self.multihead:
                x_ehr_list = []
                for i in range(len(self.classifiers)):
                    x_ehr_list.append(self.classifiers[i](x_features).squeeze(3).squeeze(3).mean(2))
                
                x_ehr = torch.stack(x_ehr_list, dim=1).squeeze()
            else:
                x_ehr = self.classifier(x_features)
                x_ehr = x_ehr.squeeze(3)
                x_ehr = x_ehr.squeeze(3)
                x_ehr = x_ehr.mean(2)
            
            x_contrastive = self.contrastive_head(x_features)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)
            
            if self.return_skips:
                return x_contrastive, x_ehr, skips
            else:
                return x_contrastive, x_ehr
        elif self.vision_ssl:
            
            # Getting the output recon
            x_rec = self.conv(x)
            
            # Before do the recon
            x_features = self.avgpool(x)
            
            x_rot = self.rotation_head(x_features).squeeze()
            x_contrastive = self.contrastive_head(x_features).squeeze()
            
            return x_rot, x_contrastive, x_rec
        elif self.classifier_ssl:
            x_features = self.avgpool(x)
            
            if self.multihead:
                x_ehr_list = []
                for i in range(len(self.classifiers)):
                    x_ehr_list.append(self.classifiers[i](x_features).squeeze(3).squeeze(3).mean(2))
                
                x_classifier = torch.stack(x_ehr_list, dim=1).squeeze()
            else:
                x_classifier = self.classifier(x_features).squeeze()
                
            return x_classifier
            
            
            
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x

    def latent(self, x):
        if isinstance(x, dict):
            x = x['image']
            
        feature_maps = []

        def hook(module, _, output):
            feature_maps.append(output)

        # hook_handle = self.model.encode_image.i3_resnet.avgpool.register_forward_hook(hook)
        hook_handle = self.avgpool.register_forward_hook(hook)

        # Forward pass through the model
        output = self(x)
        
        # Remove the hook after the forward pass to avoid redundancy
        hook_handle.remove()
        
        return feature_maps[0].reshape(feature_maps[0].size(0), -1)

def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(
            bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(
            bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        def run_function(input_x):
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            return out
        # residual = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        # if x.requires_grad:
            # out = checkpoint.checkpoint(run_function, x)
        # else:
        out = run_function(x)

        out = out + residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        inflate.inflate_batch_norm(downsample2d[1]))
    return downsample3d

def load_merlin_image_to_image(config):
    resnet = torchvision.models.resnet152(pretrained=True)
    model = MerlinImageToImage(copy.deepcopy(resnet), class_nb=30, classifier_ssl=True)
    
    checkpoint = torch.load("/dataNAS/people/akkumar/contrastive-3d/pretrained_models/i3_resnet_ssl.pt")["state_dict"]
    model_state_dict = model.state_dict()
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
    # Load the state dict and capture missing/unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
    
    # Print the missing and unexpected keys
    print("Missing keys:")
    print(missing_keys)
    print("Unexpected keys:")
    print(unexpected_keys)
    
    return model

import hydra
@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(config):
    # from Data.get_data import get_data
    # datasets, dataloaders = get_data(config)

    device = torch.device('cuda:0')

    x = torch.randn(2, 1, 224, 224, 224)
    x = x.to(device)  # Input tensor
    model = load_merlin_image_to_image(config)
    model = model.to(device)

    with torch.inference_mode():
        y = model(x)
        output = model.latent(x)
    print(output.shape)

if __name__ == "__main__":
    main()