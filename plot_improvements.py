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
from failure_case_analysis import plot_topk_images_for_class, plot_topk_images, plot_attention_map_enhance
# plot modules
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_improvements(images_1, targets_1, targets_2, predictions_1, predictions_2, similarities_1, similarities_2, classnames, dataset, model_1, model_2, preprocess, k=3):
    """
    Plot the top-k images that were misclassified in the first model and correctly classified in the second model.
    
    Args:
        images_1 (np.ndarray): Array of images from the first model (N, C, H, W)
        targets_1 (np.ndarray): Array of true labels for the first model 
        targets_2 (np.ndarray): Array of true labels for the second model
        predictions_1 (np.ndarray): Array of predicted labels for the first model
        predictions_2 (np.ndarray): Array of predicted labels for the second model
        similarities_1 (np.ndarray): Array of similarity values for the first model
        similarities_2 (np.ndarray): Array of similarity values for the second model
        classnames (list): List of class names
        dataset (Dataset): Dataset object
        model_1 (FewShotClip): Model 1 => model with lower accuracy (baseline)
        model_2 (FewShotClip): Model 2 => model with higher accuracy (improved)
        preprocess (function): Preprocessing function
        k (int): Number of top-k images to plot
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
    sorted_indices = improvement_indices[-np.argsort(improvement_deltas)]
    # Select top-k improvement indices
    top_indices = sorted_indices[:k]

    print(f"Found {len(improvement_indices)} improved images.")
    print(f"Selected indices: {top_indices}")

    # Normalize images for display
    images_display = np.transpose(images_1, (0, 2, 3, 1))
    images_display = (images_display - images_display.min()) / (images_display.max() - images_display.min())
    
    # Prepare plot
    num_rows = len(top_indices)
    fig = plt.figure(figsize=(12, 5 * num_rows))  # Adjust figure size dynamically
    gs = GridSpec(num_rows, 4, height_ratios=[1] * num_rows)

    cmap = plt.colormaps["jet"]

    # Loop over each image and plot
    for i, idx in enumerate(top_indices):
        true_label = classnames[targets_1[idx]]
        pred_label_1 = classnames[predictions_1[idx]]
        pred_label_2 = classnames[predictions_2[idx]]
        delta = improvement_deltas[i]

        # Get attention maps for both models (call plot_attention_map_enhance with plot=False to get the attention maps and not plot them) 
        attention_map_1, salient_mask_1 = plot_attention_map_enhance(dataset[idx].impath, preprocess, model_1, idx, False)
        attention_map_2, salient_mask_2 = plot_attention_map_enhance(dataset[idx].impath, preprocess, model_2, idx, False)
        attention_diff = attention_map_2 - attention_map_1

        h, w, _ = images_display[idx].shape # for resizing attention maps

        # Plot the original image
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(images_display[idx])
        ax_img.axis('off')
        ax_img.set_title(f"Original Image\nTrue: {true_label}\nDelta: {delta:.2f}")
        
        # Plot the attention map 1
        salient_heatmap = np.zeros_like(attention_map_1)
        salient_heatmap[salient_mask_1] = attention_map_1[salient_mask_1]
        salient_heatmap_resized = resize(salient_heatmap, (h, w), 
                                        order=3, mode='constant')
        ax_attn = fig.add_subplot(gs[i, 1])
        ax_attn.imshow(images_display[idx])
        ax_attn.imshow(cmap(salient_heatmap_resized), alpha=0.3, cmap=cmap)
        #ax_attn.imshow(attention_map_1)
        ax_attn.axis('off')
        ax_attn.set_title(f"Attention Map Before\nPred1: {pred_label_1}")
        
        # Plot the attention map 2 
        salient_heatmap = np.zeros_like(attention_map_2)
        salient_heatmap[salient_mask_2] = attention_map_2[salient_mask_2]
        salient_heatmap_resized = resize(salient_heatmap, (h, w), 
                                        order=3, mode='constant')
        ax_attn = fig.add_subplot(gs[i, 2])
        ax_attn.imshow(images_display[idx])
        ax_attn.imshow(cmap(salient_heatmap_resized), alpha=0.3, cmap=cmap)
        #ax_attn.imshow(attention_map_2)
        ax_attn.axis('off')
        ax_attn.set_title(f"Attention Map After\nPred2: {pred_label_2}")

        # Plot the difference in attention maps
        diff_resized = resize(attention_diff, (h, w), order=3, mode='constant')
        ax_diff = fig.add_subplot(gs[i, 3])
        ax_diff.imshow(images_display[idx])
        ax_diff.imshow(cmap(diff_resized), alpha=0.3)
        ax_diff.axis('off')
        ax_diff.set_title("Difference in\nAttention Maps")

    plt.tight_layout()
    plt.savefig("improvements.png", dpi=300)
    plt.show()


def main():
    # General settings
    args = SimpleNamespace(backbone="ViT-B/16", dataset="eurosat", root_path="data/", shots=5, batch_size=32) # edit this as needed
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
    clip_model_1, preprocess = clip.load(args.backbone)
    clip_model_2, preprocess = clip.load(args.backbone)
    clip_model_1.eval()
    clip_model_2.eval()
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
    model_1 = FewShotClip(args_1, clip_model_1).cuda()
    model_2 = FewShotClip(args_2, clip_model_2).cuda()

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
    plot_improvements(images_1, targets_1, targets_2, predictions_1, predictions_2, similarities_1, similarities_2, dataset.classnames, dataset.test, model_1, model_2, preprocess, k=3)
    
if __name__ == '__main__':
    main()