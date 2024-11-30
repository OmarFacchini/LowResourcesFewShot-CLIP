# General modules
import argparse  
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import csv
# Local modules
from datasets import build_dataset
from datasets.utils import build_data_loader
import modules.clip as clip
from modules.runner import train_model, eval_model, eval_and_get_data
from modules.utils import *
from failure_case_analysis import plot_topk_images_for_class, plot_topk_images, plot_attention_map_enhance, plot_confusion_matrix
from modules.model import FewShotClip, get_text_target_features, get_vision_target_features
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
    parser.add_argument('--n_iters', default=8, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate_LoRA', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    parser.add_argument('--dropout_rate_MetaAdapter', default=0.5, type=float, help='dropout rate applied before the MetaAdapter module')
    parser.add_argument('--bank_size', default=100, type=int, help='size of the feature bank for Breaking Loss')
    parser.add_argument('--lambda_breaking', default=0.1, type=float, help='size of the feature bank for Breaking Loss')

    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='few_shot_clip', help='file name to save the lora weights (.pt extension will be added)')
    parser.add_argument('--load_ckpt', default=None, help='Modle checkpoint to load')

    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    parser.add_argument('--plot_metrics', default=False, action='store_true', help='save features for visualization / analysis')
    parser.add_argument('--model_stats_to_csv', default=False, action='store_true', help='save features for visualization / analysis')
    # flags to add modules to CLIP
    parser.add_argument('--enable_MetaAdapter', default=False, action='store_true', help='add Meta-Adapter to CLIP model')
    parser.add_argument('--enable_lora', default=False, action='store_true', help='add LoRA adapter to CLIP model')
    parser.add_argument('--enable_BitFit', default=False, action='store_true', help='add BitFit adapter to CLIP model')
    parser.add_argument('--enable_breaking_loss', default=False, action='store_true', help='Train model with Preserving + Breaking contrastive loss')
    
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

    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    target_loader = None

    val_loader = build_data_loader(data_source=dataset.val, batch_size=args.batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8, task_type=task_type)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=args.batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8, task_type=task_type)
    if task_type == 'image2image':
        target_loader = build_data_loader(data_source=dataset.target, batch_size=args.batch_size, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8, task_type=task_type)

    train_loader = None
    if not args.eval_only:
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)

    # Prepare model
    model = FewShotClip(args, clip_model).cuda()
    model._params_to_float()
    meta_query, meta_key = None, None
    # Load model checkpoint if specified
    if args.load_ckpt is not None:
        checkpoint = torch.load(args.load_ckpt, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.enable_MetaAdapter :
            meta_query = checkpoint['meta_query'].cuda()
            meta_key = checkpoint['meta_key'].cuda()
    print("MODEL SIZE => ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Prepare class features according to modality
    model.eval()
    with torch.no_grad():
        target_features = get_text_target_features(model, dataset) if task_type == 'image2text' else get_vision_target_features(model, target_loader)

    if args.eval_only:
        print("Testing model...")
        if not args.plot_metrics :
            acc_test = eval_model(args, model, logit_scale, test_loader, target_features, meta_query=meta_query, meta_key=meta_key, support_img_loader=val_loader)
        else :
            acc_test, images, targets, predictions, features, similarities = eval_and_get_data(args, model, logit_scale, test_loader, target_features, meta_query=meta_query, meta_key=meta_key, support_img_loader=val_loader)
            print("Generating metrics...")
            if args.plot_metrics :
                k = 5
                plot_topk_images_for_class(images, targets, predictions, similarities, dataset.classnames, k=k, model=model, preprocess=preprocess, dataset=dataset.test, mode="incorrect")
                plot_topk_images_for_class(images, targets, predictions, similarities, dataset.classnames, k=k, model=model, preprocess=preprocess, dataset=dataset.test, mode="correct")
                plot_topk_images(images, targets, predictions, similarities, dataset.classnames, k=k, model=model, preprocess=preprocess, dataset=dataset.test, mode="correct")
                plot_topk_images(images, targets, predictions, similarities, dataset.classnames, k=k, model=model, preprocess=preprocess, dataset=dataset.test, mode="incorrect")

            if args.model_stats_to_csv:
                print("Saving model stats to csv...")
                model_out_to_csv(features, targets, predictions, similarities, csv_filename='results/results_csv/evaluation_results.csv')
        
        print("**** Test accuracy: {:.3f}. ****\n".format(acc_test))
    else :
        print("Training model...")
        train_model(args, model, logit_scale, dataset, train_loader, val_loader, test_loader, target_loader, target_features, task_type)

if __name__ == '__main__':
    main()