# General modules
import argparse  
import random
import numpy as np
import torch
import torchvision.transforms as transforms
# Local modules
from datasets import build_dataset
from datasets.utils import build_data_loader
import modules.clip as clip
from modules.runner import train_model, eval_model
from modules.utils import *

from modules.model import FewShotClip
#torch.autograd.set_detect_anomaly(True)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=3, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='few_shot_clip', help='file name to save the lora weights (.pt extension will be added)')
    parser.add_argument('--load_ckpt', default=None, help='Modle checkpoint to load')

    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    
    # flags to add modules to CLIP
    parser.add_argument('--enable_MetaAdapter', default=False, action='store_true', help='add Meta-Adapter to CLIP model')
    parser.add_argument('--enable_lora', default=False, action='store_true', help='add LoRA adapter to CLIP model')
    parser.add_argument('--enable_BitFit', default=False, action='store_true', help='add BitFit adapter to CLIP model')
    
    args = parser.parse_args()

    return args

def main():
    # Load config file
    args = get_arguments()
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
    task_type = 'image2text'
    if args.dataset == 'historic_maps':
        task_type = 'image2image'

    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    target_loader = None

    if args.dataset == 'imagenet':
        val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    else:
        val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8, task_type=task_type)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8, task_type=task_type)
        if task_type == 'image2image':
            target_loader = build_data_loader(data_source=dataset.target, batch_size=1, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8, task_type=task_type)
    
    train_loader = None
    if not args.eval_only:
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        if args.dataset == 'imagenet':
            train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)

    model = FewShotClip(args, clip_model, dataset, target_loader, val_loader, task_type=task_type).cuda()
    if args.load_ckpt is not None:
        checkpoint = torch.load(args.load_ckpt, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    print("MODEL SIZE => ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    if args.eval_only:
        acc_test, images, targets, predictions, similarities = eval_model(args, model, test_loader, dataset, target_loader, task_type)
        print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
        # plot_confusion_matrix(targets, predictions, dataset.classnames)
        # plot_topk_images_for_class(images, targets, predictions, similarities, dataset.classnames, 3, "correct")
        # plot_topk_images_for_class(images, targets, predictions, similarities, dataset.classnames, 3, "incorrect")
        # plot_topk_images(images, targets, predictions, similarities, dataset.classnames, 5, "correct")
        # plot_topk_images(images, targets, predictions, similarities, dataset.classnames, 5, "incorrect")
        
    else :
        train_model(args, model, logit_scale, dataset, train_loader, val_loader, test_loader, target_loader, task_type)

if __name__ == '__main__':
    main()
