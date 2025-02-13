import monai
import torch
from torch import nn
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from torchvision import transforms
from peft import LoraConfig, get_peft_model
import torchvision
import copy
from torch.nn import Parameter

import open_clip
from contrastive_3d.models.factory import create_model_and_transforms

from contrastive_3d.models import transformer, monai_densenet
from contrastive_3d.models.dynamic_network_architectures.architectures import resnet as flex_resnet
from contrastive_3d.models.inflated_convnets_pytorch.src import i3res

BIOMEDCLIP_2D = True
INFLATE = False

class Normalize3D(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize3D, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        # Assumes tensor is of shape (C, D, H, W)
        B, C, D, H, W = tensor.shape
        for d in range(D):
            for c in range(C):
                tensor[:, c, d] = (tensor[:, c, d] - self.mean[c]) / self.std[c]
        return tensor


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_biomedclip = False
        self.use_ehr = config["use_ehr"]
        self.config = config
        self.use_openclip = False

        if "openclip" in config["architecture"]:
            self.openclip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', cache_dir="/dataNAS/people/akkumar/contrastive-3d/models/")
            self.openclip.set_grad_checkpointing()
            self.use_openclip = True
            self.normalize = transforms.Normalize(
                # mean=[0.48145466, 0.4578275, 0.40821073],
                # std=[0.26862954, 0.26130258, 0.27577711],
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )

        elif "biomedclip" in config["architecture"]:
            self.use_biomedclip = True
            self.temporal_cnn = False
            self.transformer = False
            self.fully_connected = True
            self.mean_pool = False
            # This calls:
            # (1) create_model_and_transforms in factory.py in open_clip. Calls model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype) on line 251
            # (2) class CustomTextCLIP(nn.Module) in model.py in open_clip. Calls visual = TimmModel on line 119 of model.py
            # (3) class TimmModel(nn.Module) in timm_model.py in open_clip. Calls self.trunk = timm.create_model on line 74 of timm_model.py
            # (4) In /dataNAS/people/lblankem/pytorch-image-models/timm/models/_factory.py, create_model() function
            # (5) In /dataNAS/people/lblankem/pytorch-image-models/timm/models/vision_transformer.py, vit_base_patch16_224()

            self.biomedclip, _, _ = create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )

            self.biomedclip.set_grad_checkpointing()

            self.biomedclip = self.biomedclip.visual

            # print("Biomedclip model: ")
            # print(self.biomedclip)

            # 3, 768, kernel_size=(16, 16), stride=(16, 16)

            # find conv layer
            if INFLATE:
                conv2d = self.biomedclip.trunk.patch_embed.proj

                conv3d = torch.nn.Conv3d(
                    3,
                    768,
                    (16, 16, 16),
                    padding=(0, 0, 0),
                    dilation=(1, 1, 1),
                    stride=(16, 16, 16)
                )

                weight_2d = conv2d.weight.data
                weight_3d = torch.zeros(*weight_2d.shape)
                weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, 16, 1, 1)
                middle_idx = 16 // 2
                weight_3d[:, :, middle_idx, :, :] = weight_2d

                conv3d.weight = Parameter(weight_3d)
                conv3d.bias = conv2d.bias

                self.biomedclip.trunk.patch_embed.proj = conv3d

            with open(
                "/dataNAS/people/akkumar/contrastive-3d/contrastive_3d/models/biomedclip.txt",
                "w",
            ) as f:
                print(self.biomedclip, file=f)

            # config = LoraConfig(
            #     r=16,
            #     lora_alpha=16,
            #     target_modules=["qkv", "fc1", "fc2", "proj"],
            #     lora_dropout=0.1,
            #     bias="none",
            #     modules_to_save=["classifier"],
            # )

            #self.biomedclip = get_peft_model(self.biomedclip, config)
            print_trainable_parameters(self.biomedclip)

            # for param in self.biomedclip.parameters():
            #     param.requires_grad = False

            # trainable_layers = [
            #     'temporal_attn', 'temporal_ls1', 'temporal_drop_path1',
            #     'temporal_norm2', 'temporal_mlp', 'temporal_ls2', 'temporal_drop_path2'
            # ]

            # for name, module in self.biomedclip.named_modules():
            #     if any(trainable_layer in name for trainable_layer in trainable_layers):
            #         print("Unfreezing", name)
            #         sys.stdout.flush()
            #         for param in module.parameters():
            #             param.requires_grad = True

            self.encode_image_2d = nn.Sequential(
                nn.Linear(
                    512, 512
                ),  # Assuming visual embeddings have a method called embed_dim
            )

            # freeze all parameters in biomedclip
            # for param in self.biomedclip.parameters():
            #   param.requires_grad = False
            self.normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            self.normalize_3d = Normalize3D(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            if self.temporal_cnn:
                self.temporal_transform = nn.Sequential(
                    nn.Conv1d(152, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(256, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(128, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            elif self.transformer:
                self.positional_embedding = nn.Parameter(torch.empty(160, 512))
                self.transformer = transformer.Transformer(
                    width=512,
                    layers=2,
                    heads=4,
                )
                self.ln_final = transformer.LayerNorm(512)
            elif self.fully_connected:
                self.fc = nn.Sequential(
                    nn.Linear(512 * 160, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                )
        else:
            if "densenet" in config["architecture"]:
                dense_net = monai_densenet.densenet121(
                    spatial_dims=3, in_channels=1, out_channels=1692, gradient_checkpointing = True
                )
                # dense_net.load_state_dict(
                #     torch.load(
                #         "/dataNAS/people/lblankem/contrastive-3d/models/densenet_best_phecode_12-06-2023_13-48-08.pt"
                #     )
                # )
                self.dense_net = dense_net
                # self.features = dense_net.features
                self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.encode_image = nn.Sequential(
                    nn.Linear(1024, 512),
                )
            # elif "resnet" in config["architecture"]:
            #     self.resnet = flex_resnet.ResNet152(n_classes=1692, n_input_channels=1, input_dimension=3, squeeze_excitation=True)
            #     # self.resnet.load_state_dict(
            #     #     torch.load(
            #     #         "/dataNAS/people/lblankem/contrastive-3d/models/resnet_best_phecode_12-18-2023_12-17-24.pt"
            #     #     )
            #     # )
            #     self.encode_image = nn.Sequential(
            #         nn.Linear(512, 512),
            #     )

            elif "i3_resnet" in config["architecture"]:
                resnet = torchvision.models.resnet152(pretrained=True)
                model = i3res.I3ResNet(copy.deepcopy(resnet), class_nb=1692, conv_class=True)
                # checkpoint = torch.load("/dataNAS/people/lblankem/contrastive-3d/models/i3resnet_best_phecode_01-11-2024_20-34-15.pt")
                try:
                    checkpoint = torch.load("/dataNAS/people/akkumar/contrastive-3d/models/i3resnet_best_phecode_01-23-2024_14-19-22_epoch_15.pt")
                    # checkpoint = torch.load("/dataNAS/people/lblankem/contrastive-3d/models/i3resnet_best_phecode_02-18-2024_17-18-55_epoch_8.pt")
                    model_state_dict = model.state_dict()
                    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
                    model.load_state_dict(filtered_checkpoint, strict=False)
                except:
                    pass
                self.i3_resnet = model


    def forward(self, image):
        if self.use_openclip:
            batch, _, height, width, slices = image.size()
            image = torch.flip(image, dims=[2, 3])
            image = torch.transpose(image, 2, 3)
            image_2d = image.reshape(batch * slices, 1, height, width)
            image_2d = image_2d.repeat(1, 3, 1, 1)
            image_2d = self.normalize(image_2d)
            slice_features = self.openclip.encode_image(image_2d)
            slice_features_reshaped = slice_features.view(batch, slices, -1)
            ehr_embeddings = torch.zeros((batch, 1692)).cuda()
            return slice_features_reshaped, ehr_embeddings

        if self.use_biomedclip:
            batch, _, height, width, slices = image.size()
            image = torch.flip(image, dims=[2, 3])
            image = torch.transpose(image, 2, 3)
            if INFLATE:
                image = image.permute(0, 1, 4, 2, 3)
                image_2d = image.repeat(1, 3, 1, 1, 1)
                image_2d = self.normalize_3d(image_2d)
            else:
                image = image.permute(0, 1, 4, 2, 3)
                image = image.reshape(batch * slices, 1, height, width)
                image_2d = image.repeat(1, 3, 1, 1)
                image_2d = self.normalize(image_2d)
            slice_features = self.biomedclip(image_2d)
            # print("Slice features shape: ")
            # print(slice_features.shape)
            # sys.stdout.flush()
            ehr_embeddings = torch.zeros((batch, 1692)).cuda()
            if INFLATE:
                return slice_features, ehr_embeddings
            # slice_features_reshaped = slice_features_reshaped.mean(dim=1)
            if BIOMEDCLIP_2D:
                slice_features_reshaped = slice_features.view(batch, slices, -1)
                return (slice_features_reshaped, ehr_embeddings)
            if self.temporal_cnn:
                temporal_features = self.temporal_transform(slice_features_reshaped)
                avg_features = temporal_features.mean(dim=1)
                avg_features = avg_features.squeeze()
            else:
                if self.transformer:
                    x = slice_features_reshaped
                    x = x + self.positional_embedding[:152]
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.transformer(x)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    avg_features = x.mean(dim=1)
                    avg_features = self.ln_final(avg_features)
                    avg_features = self.encode_image_2d(avg_features)
                    return avg_features
                elif self.fully_connected:
                    x = slice_features.view(batch, slices, -1)
                    x = x.view(batch, -1)
                    x = self.fc(x)
                    return x, ehr_embeddings
                elif self.mean_pool:
                    x = slice_features_reshaped
                    x = x.mean(dim=1)
                    x = x.squeeze()
                    return x
                else:
                    x = slice_features_reshaped
                    x = x.mean(dim=1)
                    x = x.squeeze()
                    return x
        elif "densenet" in self.config["architecture"]:
            ehr_output = self.dense_net(image)
            image_features = self.dense_net.features(image)
            image_features = self.adaptive_pool(image_features)
            image_features = image_features.squeeze()
            image_features = self.encode_image(image_features)
            return image_features, ehr_output
        
        # elif "resnet" in self.config["architecture"]:
        #     image_features, ehr_output = self.resnet(image)
        #     image_features = self.encode_image(image_features)
        #     return image_features, ehr_output

        elif "i3_resnet" in self.config["architecture"]:

            contrastive_features, ehr_features, skips = self.i3_resnet(image)
            # # image_features = self.encode_image(image_features)
            # return contrastive_features, ehr_features, skips
            return contrastive_features, ehr_features
            # contrastive_features, ehr_features = self.i3_resnet(image)
            # # image_features = self.encode_image(image_features)
            # return contrastive_features, ehr_features


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if "pubmedbert" in config["text_encoder"]:
            self.text_encoder = AutoModel.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            )
            self.text_encoder = resize_text_pos_embed(self.text_encoder)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            )
        elif "longformer" in config["text_encoder"]:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "yikuan8/Clinical-Longformer"
            )
            self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
            self.text_encoder.gradient_checkpointing_enable()
        elif "biomedclip" in config["text_encoder"]:
            self.tokenizer = open_clip.get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self.biomedclip, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self.biomedclip.set_grad_checkpointing()
            self.biomedclip = self.biomedclip.text
            # freeze all parameters in biomedclip
            # for param in self.biomedclip.parameters():
            #     param.requires_grad = False
        elif "openclip" in config["text_encoder"]:
            self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
            self.openclip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', cache_dir="/dataNAS/people/akkumar/contrastive-3d/models/")
            self.openclip.set_grad_checkpointing()
            # self.openclip = self.openclip
        else:
            raise ValueError("Invalid text encoder.")

        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        # print("Text labels:")
        # for text in text_labels:
        #     print(text)
        # print()
        # sys.stdout.flush()
        if isinstance(text_labels[0], str):
            text_labels = [text.lower() for text in text_labels]
            if "openclip" in self.config["text_encoder"]:
                inputs = self.tokenizer(
                    text_labels,
                )
            elif "biomedclip" in self.config["text_encoder"]:
                inputs = self.tokenizer(
                    text_labels,
                    context_length=1024,
                )
            else:
                inputs = self.tokenizer(
                    text_labels,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                )
            inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}

        else:
            inputs = {'input_ids': text_labels[0].to(self.text_encoder.device), 'attention_mask': torch.ones((1, text_labels[0].shape[1])).to(self.text_encoder.device)}

        if "pubmedbert" in self.config["text_encoder"]:
            text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        elif "longformer" in self.config["text_encoder"]:
            text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
            text_embeddings = self.linear_layer(text_embeddings)
        elif "openclip" in self.config["text_encoder"]:
            inputs = inputs.cuda()
            text_embeddings = self.openclip.encode_text(inputs)
            # text_embeddings = text_embeddings.mean(dim=1)
            # text_embeddings = text_embeddings.squeeze()
            return text_embeddings
        elif BIOMEDCLIP_2D:
            inputs = inputs.cuda()
            print("INPUTS SHAPE")
            print(inputs.shape)
            sys.stdout.flush()
            inputs = inputs.view(-1, 256).cuda()
            text_embeddings = self.biomedclip(inputs).view(-1, 4, 512)
            text_embeddings = text_embeddings.mean(dim=1)
            # text_embeddings = text_embeddings[:, 0, :]
            text_embeddings = text_embeddings.squeeze()
            return text_embeddings
        elif "biomedclip" in self.config["text_encoder"]:
            print("INPUTS SHAPE")
            print(inputs.shape)
            sys.stdout.flush()
            inputs = inputs.view(-1, 256).cuda()
            text_embeddings = self.biomedclip(inputs).view(-1, 4, 256)
            text_embeddings = text_embeddings.mean(dim=1)
            # text_embeddings = text_embeddings[:, 0, :]
            text_embeddings = text_embeddings.squeeze()
        return text_embeddings


class Clip3D(nn.Module):
    def __init__(
        self, config, init_logit_scale: float = 1.0, init_logit_bias: float = 0.0
    ):
        super().__init__()
        self.encode_image = ImageEncoder(config)
        self.encode_text = TextEncoder(config)
        self.config = config
        # self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        # self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        self.logit_bias = None

    def forward(self, image, text):
        # if self.config["use_ehr"]:
        image_features, ehr_features = self.encode_image(image)
        # else:
        #     image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # print("Check 2d biomedclip: ")
        # print(image_features.shape)
        # print(text_features.shape)

        # if there is just one dimension, add a batch dimension
        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if ("biomedclip" in self.config["text_encoder"]) or ("openclip" in self.config["text_encoder"]):
            return (
                image_features,
                ehr_features,
                text_features,
            )


        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        if self.config['use_ehr']:
            return (
                image_features,
                ehr_features,
                text_features,
                logits_per_image,
                logits_per_text,
                self.logit_scale.exp(),
                self.logit_bias,
            )
        else:
            # create ehr features as a dummy tensor
            batchsize = image_features.shape[0]
            ehr_features = torch.zeros((batchsize, 1692)).cuda()
            return (
                image_features,
                ehr_features,
                text_features,
                logits_per_image,
                logits_per_text,
                self.logit_scale.exp(),
                self.logit_bias,
            )


def resize_text_pos_embed(text_encoder):
    print(text_encoder.embeddings)
    original_embeddings = text_encoder.embeddings.position_embeddings.weight
    original_num_positions, embedding_size = original_embeddings.size()
    new_num_positions = original_num_positions * 2
    new_embeddings = torch.nn.Embedding(new_num_positions, embedding_size).to(
        original_embeddings.device
    )
    for new_pos in range(new_num_positions):
        if new_pos % 2 == 0:
            original_pos = new_pos // 2
            new_embeddings.weight.data[new_pos] = original_embeddings.data[original_pos]
        else:
            lower = new_pos // 2
            upper = min(lower + 1, original_num_positions - 1)
            new_embeddings.weight.data[new_pos] = (
                0.5 * original_embeddings.data[lower]
                + 0.5 * original_embeddings.data[upper]
            )
    text_encoder.embeddings.position_embeddings = new_embeddings
    text_encoder.config.max_position_embeddings = new_num_positions
    text_encoder.embeddings.register_buffer(
        "position_ids", torch.arange(1024).expand((1, -1)), persistent=False
    )
    text_encoder.embeddings.register_buffer(
        "token_type_ids",
        torch.zeros(text_encoder.embeddings.position_ids.size(), dtype=torch.long),
        persistent=False,
    )
    print(text_encoder.embeddings)
    return text_encoder


def resize_vision_pos_embed(embed_layer, new_num_positions):
    original_embeddings = embed_layer.weight
    original_num_positions, embedding_size = original_embeddings.size()
    new_embeddings = torch.nn.Embedding(new_num_positions, embedding_size).to(
        original_embeddings.device
    )
    for new_pos in range(new_num_positions):
        if new_pos % 2 == 0:
            original_pos = new_pos // 2
            new_embeddings.weight.data[new_pos] = original_embeddings.data[original_pos]
        else:
            lower = new_pos // 2
            upper = min(lower + 1, original_num_positions - 1)
            new_embeddings.weight.data[new_pos] = (
                0.5 * original_embeddings.data[lower]
                + 0.5 * original_embeddings.data[upper]
            )
    embed_layer.weight = new_embeddings
    return embed_layer
