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
from .meta_adapter import *
# CLIP
from .clip import *

def edit_clip_model(args, clip_model, test_only=False):
    """
    Edit CLIP model with chosen modules (LoRA, Meta-Adapter, etc.)
    (currently is a Copy from apply_lora from LoRa-challengingDatasets/modules/lora/lora.py)
    """
    list_metaAdapter_layers = None
    list_lora_layers = None
    list_bitfit_parameters = None

    print("Turning off all gradients for CLIP model.")
    for p in clip_model.parameters():
        if p.requires_grad:
            p.requires_grad = False

    if args.enable_BitFit and not test_only:
        print("Adding BitFit to CLIP model. Biases are trained.")
        list_bitfit_parameters = []
        # Turn on gradients for all biases
        for n, p in clip_model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
                list_bitfit_parameters.append(p)

    if args.enable_lora:
        print("Adding LoRA to CLIP model.")
        list_lora_layers = apply_lora(args, clip_model)
        # Turn on gradients for all LoRA layers
        if test_only:
            for n, p in clip_model.named_parameters():
                if 'lora_' in n:
                    p.requires_grad = False

    if args.enable_MetaAdapter:
        print("Adding Meta-Adapter to CLIP model.")
        pass
        #list_metaAdapter_layers = apply_MetaAdapter(args, clip_model)
    
    return list_metaAdapter_layers, list_lora_layers, list_bitfit_parameters


def evaluate_model(args, clip_model, logit_scale, loader, dataset, task_type):
    # Load Model on GPU
    clip_model = clip_model.cuda()
    # Run Inference
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def train_model(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, task_type):
    """
    Run CLIP with chosen modules (LoRA, Meta-Adapter, etc.)
    (currently is a Copy from run_lora from LoRa-challengingDatasets/modules/lora/lora.py)
    """
    VALIDATION = False
    # Add chosen modules to CLIP
    list_metaAdapter_layers, list_lora_layers, list_bitfit_parameters = edit_clip_model(args, clip_model)
    # Load Model on GPU
    clip_model = clip_model.cuda()

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
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
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
            acc_val = evaluate_model(args, clip_model, logit_scale ,val_loader, dataset, task_type)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
    

    acc_test = evaluate_model(args, clip_model, logit_scale ,test_loader, dataset, task_type)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
        #save_BitFit(args, list_bitfit_parameters)
        #save_metaAdapter(args, list_metaAdapter_layers)
        
    return
