# General
import torch
import torch.nn.functional as F

# Local Modules
from .utils import *
# LoRA moduels
from .lora import *
from .lora.loralib.utils import apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from .lora.loralib import layers as lora_layers
# BitFit
from .bitfit.bitfit import apply_BitFit
# Meta-Adapter
from .meta_adapter.meta_adapter import MetaAdapter
# CLIP
from .clip import *

def edit_clip_model(args, clip_model, test_only=False):
    """
    Edit CLIP model with chosen modules (LoRA, Meta-Adapter, etc.)
    (currently is a Copy from apply_lora from LoRa-challengingDatasets/modules/lora/lora.py)
    """
    list_metaAdapter_layers = []
    list_lora_layers = []
    list_bitfit_parameters = []

    print("Turning off all gradients for CLIP model.")
    for p in clip_model.parameters():
        if p.requires_grad:
            p.requires_grad = False
    print("Trainable parameters => ", sum(p.numel() for p in clip_model.parameters() if p.requires_grad))

    if args.enable_BitFit and not test_only:
        print("Adding BitFit to CLIP model. Biases are trained.")
        list_bitfit_parameters = []
        # Turn on gradients for all biases
        for n, p in clip_model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
                list_bitfit_parameters.append(p)
        print("     Trainable Size => ", sum(p.numel() for p in clip_model.parameters() if p.requires_grad))

    if args.enable_lora:
        print("Adding LoRA to CLIP model.")
        list_lora_layers = apply_lora(args, clip_model)
        # Turn on gradients for all LoRA layers
        if test_only:
            print("     Turning off gradients for LoRA layers.")
            for n, p in clip_model.named_parameters():
                if 'lora_' in n:
                    p.requires_grad = False
        print("     Trainable Size => ", sum(p.numel() for p in clip_model.parameters() if p.requires_grad))

    
    return list_lora_layers, list_bitfit_parameters


def get_text_labels_features(clip_model, dataset):
    """
    Parse through Text encoder text class features, normalize and return
    """
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            text_embedding = clip_model.encode_text(texts)
        text_features = text_embedding/text_embedding.norm(dim=-1, keepdim=True)
    return text_features

def get_vision_labels_features(clip_model, loader):
    """
    Parse through Vision encoder vision class features, normalize and return
    """
    # Data augmentation for the cache model
    features_list = []
    for i, (images, target) in enumerate(tqdm(loader)):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(images)
            features_list.append(image_features)

    vision_features = torch.cat(features_list, dim=0).mean(dim=0)    
    vision_features /= vision_features.norm(dim=-1, keepdim=True)

    return vision_features


def eval_model(args, clip_model, loader, dataset, task_type):
    """
    Zero-shot evaluation of CLIP model
    """
    # Load Model on GPU
    clip_model = clip_model.cuda()
    clip_model.eval()
    # Prepare class features according to modality
    class_features = get_text_labels_features(clip_model, dataset) if task_type == 'image2text' else get_vision_labels_features(clip_model, [])

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ class_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def train_model(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, task_type):
    """
    Run CLIP with chosen modules (LoRA, Meta-Adapter, etc.)
    (currently is a Copy from run_lora from LoRa-challengingDatasets/modules/lora/lora.py)
    """
    VALIDATION = True
    # Add chosen modules to CLIP
    list_lora_layers, list_bitfit_parameters = edit_clip_model(args, clip_model)
    # Load Model on GPU
    clip_model = clip_model.cuda()
    # Precompute class features
    clip_model = clip_model.eval()
    target_features = get_text_labels_features(clip_model, dataset) if task_type == 'image2text' else get_vision_labels_features(clip_model, [])

    # Load meta-adapter
    if args.enable_MetaAdapter:
        meta_adapter = MetaAdapter(dim=target_features.shape[-1]).to(clip_model.dtype).cuda()
        print("Adding Meta-Adapter to CLIP model.")
        print("     Trainable Size => ", sum(p.numel() for p in clip_model.parameters() if p.requires_grad) + sum(p.numel() for p in meta_adapter.parameters() if p.requires_grad))
        # Obtain support features
        support_features = target_features
        # Category embeddings
        query = target_features
        # Support embedding
        key = support_features.reshape(query.shape[0], -1, query.shape[1])

    # Set up optimizer and scheduler
    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    # training model
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.

        for i, (images, target) in enumerate(tqdm(train_loader)):  
            # Load data on GPU
            images, target = images.cuda(), target.cuda()
            # Load Label features
            if  (args.enable_lora and task_type == 'image2text' and (args.encoder == 'text' or args.encoder == 'both')) or \
                (args.enable_lora and task_type == 'image2image' and (args.encoder == 'vision' or args.encoder == 'both')) or \
                (args.enable_BitFit) :
                target_features = get_text_labels_features(clip_model, dataset) if task_type == 'image2text' else get_vision_labels_features(clip_model, [])
            # Forward the batch
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            # Forward through Meta-Adapter
            if args.enable_MetaAdapter:
                meta_adaptation = meta_adapter(query, key, key)
                cosine_similarity = logit_scale * image_features @ meta_adaptation.T
            else :
                # directly get similarity scores with class features
                cosine_similarity = logit_scale * image_features @ target_features.T
            
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = eval_model(args, clip_model, val_loader, dataset, task_type)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
    
    acc_test = eval_model(args, clip_model, test_loader, dataset, task_type)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
        #save_BitFit(args, list_bitfit_parameters)
        #save_metaAdapter(args, list_metaAdapter_layers)
        
    return
