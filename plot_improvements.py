# General modules
import random
import numpy as np
import torch
from types import SimpleNamespace
# Local modules
from datasets import build_dataset
from datasets.utils import build_data_loader
import modules.clip as clip
from modules.runner import train_model, eval_model, eval_and_get_data
from modules.utils import *
from modules.model import FewShotClip, get_text_target_features, get_vision_target_features
# plot modules
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#TODO: add plotting of attention maps


def plot_improvements(images_1, targets_1, targets_2, predictions_1, predictions_2, similarities_1, similarities_2, classnames, k=3):
    """
    Plot the top-k images that were misclassified in the first model and correctly classified in the second model.
    
    Args:
        images_1 (np.ndarray): Array of images from the first model (N, C, H, W)
        targets_1 (np.ndarray): Array of true labels for the first model
        targets_2 (np.ndarray): Array of true labels for the second model
        predictions_1 (np.ndarray): Array of predicted labels for the first model
        predictions_2 (np.ndarray): Array of predicted labels for the second model
        similarities_1 (np.ndarray): Array of cosine similarities for the first model
        similarities_2 (np.ndarray): Array of cosine similarities for the second model
        classnames (list): List of class names
        k (int): Number of top images to consider
    """
    
    # Identify indices of misclassified in model 1 and correctly classified in model 2
    improvement_indices = np.where((predictions_1 != targets_1) & (predictions_2 == targets_2))[0]
    
    if len(improvement_indices) == 0:
        print("No improvements found between the models.")
        return
    
    # take similarity values for the true labels
    similarities_1 = similarities_1[np.arange(len(targets_1)), targets_1] 
    similarities_2 = similarities_2[np.arange(len(targets_2)), targets_2]

    # compute delta similarities
    delta_similarities = similarities_2 - similarities_1 
    improvement_deltas = delta_similarities[improvement_indices] # get deltas for the improvement indices

    # Sort the indices based on the improvement so larger deltas are first
    sorted_indices = improvement_indices[np.argsort(-improvement_deltas)]
    # Select top-k improvement indices
    top_indices = sorted_indices[:k]

    print(f"Found {len(improvement_indices)} improved images.")
    print(f"Selected indices: {top_indices}")

    # Normalize images for display
    images_display = np.transpose(images_1, (0, 2, 3, 1))
    images_display = (images_display - images_display.min()) / (images_display.max() - images_display.min())
    
    # Plot the top-k images using GridSpec
    fig = plt.figure(figsize=(15, 7))  # Increase the height for title space
    gs = GridSpec(2, len(top_indices), height_ratios=[0.1, 1])  # 1 row for title, 1 row for images

    # Add space for the title in the top row
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')  # Hide axis for title subplot
    title_ax.text(0.5, 0.5, "Top-k Improvements", fontsize=16, ha='center', va='center')

    # Loop over each image and plot in the bottom row
    for i, idx in enumerate(top_indices):
        true_label = classnames[targets_1[idx]]
        pred_label_1 = classnames[predictions_1[idx]]
        pred_label_2 = classnames[predictions_2[idx]]
        delta = improvement_deltas[i]
        print(delta)
        
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(images_display[idx])
        ax.axis('off')
        ax.set_title(f"True: {true_label}\nPred1: {pred_label_1}\nPred2: {pred_label_2}\nDelta: {delta:.2f}")

    plt.tight_layout()
    plt.savefig("improvements.png")
    plt.show()

    return top_indices


def main():
    # General settings
    args = SimpleNamespace(backbone="ViT-B/16", dataset="eurosat", root_path="data/", shots=5, batch_size=32)
    # Model 1
    args_1 = SimpleNamespace(
        enable_BitFit=False, enable_lora=True, enable_MetaAdapter=False, enable_breaking_loss=False, eval_only=True, # edit this as needed
        backbone="ViT-B/16", dataset="eurosat", root_path="data/", shots=5, batch_size=32, load_ckpt="models/lora.pth", # edit this as needed
        position='all', encoder='both', params=['q', 'k', 'v'], r=2, alpha=1, dropout_rate_LoRA=0.25, dropout_rate_MetaAdapter=0.5, bank_size=100, lambda_breaking=0.1
    )
    # Model 2
    args_2 = SimpleNamespace(
        enable_BitFit=True, enable_lora=True, enable_MetaAdapter=True, enable_breaking_loss=False, eval_only=True, # edit this as needed
        backbone="ViT-B/16", dataset="eurosat", root_path="data/", shots=5, batch_size=32, load_ckpt="models/lora_bitfit_meta.pth", # edit this as needed
        position='all', encoder='both', params=['q', 'k', 'v'], r=2, alpha=1, dropout_rate_LoRA=0.25, dropout_rate_MetaAdapter=0.5, bank_size=100, lambda_breaking=0.1
    )

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

    # Prepare model
    model_1 = FewShotClip(args_1, clip_model).cuda()
    model_2 = FewShotClip(args_2, clip_model).cuda()

    meta_query = None
    meta_key = None

    # load model 1 
    if args_1.load_ckpt is not None:
        checkpoint = torch.load(args_1.load_ckpt, weights_only=True)
        model_1.load_state_dict(checkpoint['model_state_dict'])
        model_1 = model_1.float()
        if args_1.enable_MetaAdapter :
            meta_query = checkpoint['meta_query'].cuda()
            meta_key = checkpoint['meta_key'].cuda()

    # load model 2
    if args_2.load_ckpt is not None:
        checkpoint = torch.load(args_2.load_ckpt, weights_only=True)
        model_2.load_state_dict(checkpoint['model_state_dict'])
        model_2 = model_2.float()
        if args_2.enable_MetaAdapter :
            meta_query = checkpoint['meta_query'].cuda()
            meta_key = checkpoint['meta_key'].cuda()


    # Prepare class features according to modality
    with torch.no_grad():
        model_1 = model_1.eval()
        target_features_1 = get_text_target_features(model_1, dataset) if task_type == 'image2text' else get_vision_target_features(model_1, target_loader)
        model_2 = model_2.eval()
        target_features_2 = get_text_target_features(model_2, dataset) if task_type == 'image2text' else get_vision_target_features(model_2, target_loader)

    print("Testing model 1...")
    acc_test_1, images_1, targets_1, predictions_1, features_1, similarities_1 = eval_and_get_data(args_1, model_1, logit_scale, test_loader, target_features_1, meta_query=meta_query, meta_key=meta_key, support_img_loader=val_loader)
    print(f"Accuracy for model 1: {acc_test_1:.2f}")
    
    print("Testing model 2...")
    acc_test_2, images_2, targets_2, predictions_2, features_2, similarities_2 = eval_and_get_data(args_2, model_2, logit_scale, test_loader, target_features_2, meta_query=meta_query, meta_key=meta_key, support_img_loader=val_loader)
    print(f"Accuracy for model 2: {acc_test_2:.2f}")
   
    print("Generating metrics...")
    top_indices = plot_improvements(images_1, targets_1, targets_2, predictions_1, predictions_2, similarities_1, similarities_2, dataset.classnames, k=3)

if __name__ == '__main__':
    main()