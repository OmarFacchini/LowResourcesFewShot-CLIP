import torch
import torch.nn.functional as F
from torch import nn

from .lora.loralib import apply_lora
from .clip import *
from .meta_adapter.meta_adapter import MetaAdapter

def get_text_target_features(model, dataset):
    """
    Parse through Text encoder text class features, normalize and return
    """
    template = dataset.template[0] 
    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        texts = clip.tokenize(texts).cuda()
        text_embedding = model.encode_text(texts)
        text_features = text_embedding/text_embedding.norm(dim=-1, keepdim=True)
    return text_features

def get_vision_target_features(model, loader):
    """
    Parse through Vision encoder vision class features, normalize and return
    """
    # Data augmentation for the cache model
    features_list = []
    for i, (images, target, breaking_img1, breaking_img2) in enumerate(loader):
        images = images.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = model.encode_image(images)
            features_list.append(image_features)
    #with torch.amp.autocast(device_type="cuda", dtype=dtype_autocast):
    vision_features = torch.cat(features_list, dim=0)    
    normalized_vision_features = vision_features/vision_features.norm(dim=-1, keepdim=True)

    return normalized_vision_features


class FewShotClip(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.visual = self.clip_model.visual
        self.encode_text = self.clip_model.encode_text
        self.encode_image = self.clip_model.encode_image
        self.encode_image_attention = self.clip_model.encode_image_attention

        # Turn off gradients for CLIP model
        print("Turning off all gradients for CLIP model.")
        for p in self.clip_model.parameters():
            if p.requires_grad:
                p.requires_grad = False

        # Add LoRA to CLIP model
        if args.enable_lora:
            print("Adding LoRA to CLIP model.")
            list_lora_layers = apply_lora(args, self.clip_model)
            # Turn on gradients for all LoRA layers
            #if args.eval_only:
            #    print("     Turning off gradients for LoRA layers.")
            #    for n, p in self.clip_model.named_parameters():
            #        if 'lora_' in n:
            #            p.requires_grad = False
        
        # Add BitFit to CLIP model
        if args.enable_BitFit and not args.eval_only:
            print("Adding BitFit to CLIP model. Biases are trained.")
            # Turn on gradients for all biases
            for n, p in self.clip_model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True

        # Load meta-adapter
        if args.enable_MetaAdapter:
            self.meta_adapter = MetaAdapter(dim=512, dropout_prob=args.dropout_rate_MetaAdapter).to(self.clip_model.dtype)
            print("Adding Meta-Adapter to CLIP model.")

    def _params_to_float(self):
        for param in self.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        
        