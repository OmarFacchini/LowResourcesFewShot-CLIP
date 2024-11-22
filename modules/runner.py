# General
import torch
import torch.nn.functional as F
import os

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
from .model import get_text_labels_features, get_vision_labels_features


def eval_model(args, model, loader, dataset, target_loader, task_type):
    """
    Zero-shot evaluation of CLIP model
    """
    # Load Model on GPU
    model.eval()
    target_features = get_text_labels_features(model, dataset) if task_type == 'image2text' else get_vision_labels_features(model, target_loader)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target, target_f) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = model.encode_image(images)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ target_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def train_model(args, model, logit_scale, dataset, train_loader, val_loader, test_loader, target_loader, task_type):
    """
    Run CLIP with chosen modules (LoRA, Meta-Adapter, etc.)
    (currently is a Copy from run_lora from LoRa-challengingDatasets/modules/lora/lora.py)
    """
    VALIDATION = True

    # Prepare class features according to modality
    model = model.eval()
    target_features = get_text_labels_features(model, dataset) if task_type == 'image2text' else get_vision_labels_features(model, target_loader)

    # Extract Query - Key pairs for Meta-Adapter
    if args.enable_MetaAdapter:
        model = model.eval()
        support_features = get_vision_labels_features(model, val_loader)
        # Category embeddings
        meta_query = target_features
        # Support embedding
        meta_key = support_features.reshape(meta_query.shape[0], -1, meta_query.shape[1])
        # Detach tensors from computation graph
        meta_query = meta_query.detach()
        meta_key = meta_key.detach()

    # Set up optimizer and scheduler
    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)

    # training model
    scaler = torch.amp.GradScaler()
    count_iters = 0
    finish = False
    
    while count_iters < total_iters:
        model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        
        train_loader_tqdm = tqdm(train_loader)
        for i, (images, target, target_f) in enumerate(train_loader_tqdm):
            train_loader_tqdm.set_description(f'Itr {count_iters}/{total_iters}')
            
            # Load data on GPU
            images, target = images.cuda(), target.cuda()
            # Load Label features
            if  (args.enable_lora and task_type == 'image2text' and (args.encoder == 'text' or args.encoder == 'both')) or \
                (args.enable_lora and task_type == 'image2image' and (args.encoder == 'vision' or args.encoder == 'both')) or \
                (args.enable_BitFit) :
                target_features = get_text_labels_features(model, dataset) if task_type == 'image2text' else get_vision_labels_features(model, target_loader)
            # Forward the batch
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_encoding = model.encode_image(images)
                    image_features = image_encoding/image_encoding.norm(dim=-1, keepdim=True)

            if args.enable_MetaAdapter:
                # Forward through Meta-Adapter
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    meta_adaptation = model.meta_adapter(meta_query, meta_key, meta_key)
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
            model.eval()
            acc_val = eval_model(args, model, val_loader, dataset, target_loader, task_type)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
    
    acc_test = eval_model(args, model, test_loader, dataset, target_loader, task_type)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        full_path = os.path.join(args.save_path, str(args.filename) + '.pth')
        torch.save({'model_state_dict':model.state_dict()}, full_path)
        print("Model saved => ", full_path)
        
    return
