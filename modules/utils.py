from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import modules.clip as clip

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    if cfg['load_cache'] == False:
        cache_keys = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")

    return cache_keys


def pre_load_features(cfg, split, clip_model, loader):
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                     range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                      range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

# ======================== 
# Plotting functions

def plot_confusion_matrix(targets, predictions, classnames):
    '''
    Plot confusion matrix with improved handling of long class names
    
    Args:
        targets: True labels
        predictions: Predicted labels
        classnames: List of class names
    '''
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Create figure with adjusted size based on number of classes
    n_classes = len(classnames)
    plt.figure(figsize=(max(8, n_classes * 0.8), max(8, n_classes * 0.8)))
    
    # Shorten class names if they're too long
    shortened_classnames = []
    for name in classnames:
        if len(name) > 20:  # Adjust threshold as needed
            # Keep first and last few characters
            shortened_name = name[:10] + '...' + name[-7:]
            shortened_classnames.append(shortened_name)
        else:
            shortened_classnames.append(name)
    
    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=shortened_classnames)
    
    # Plot with customization
    disp.plot(
        xticks_rotation=45,  # Rotate labels 45 degrees
        values_format='.0f',  # Show absolute numbers without decimals
        # cmap='Blues',        # Use Blues colormap for better readability
    )
    
    # Adjust label properties
    plt.xticks(fontsize=8, ha='right')  # Align rotated labels to the right
    plt.yticks(fontsize=8)
    
    # Add title with padding
    plt.title('Confusion Matrix', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure with high DPI
    plt.savefig('plot/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_topk_images_for_class(images, targets, predictions, similarities, classnames, k=3, mode="correct"):
    """
    Plot top-k images for each class based on correctness and similarity.
    
    Args:
        images (np.ndarray): Array of images (N, C, H, W)
        targets (np.ndarray): Array of true labels
        predictions (np.ndarray): Array of predicted labels
        similarities (np.ndarray): Array of cosine similarities
        classnames (list): List of class names
        k (int): Number of top images to consider per class
        mode (str): "correct" for correctly classified, "incorrect" for misclassified
    """
    assert mode in {"correct", "incorrect"}, "Mode must be 'correct' or 'incorrect'."
    
    # Convert images from (N, C, H, W) to (N, H, W, C) and normalize for display
    images_display = np.transpose(images, (0, 2, 3, 1))
    images_display = (images_display - images_display.min()) / (images_display.max() - images_display.min())
    
    # Determine mask and title based on mode
    if mode == "correct":
        mask = targets == predictions
        title = "Top Correctly Classified Images by Cosine Similarity"
    else:
        mask = targets != predictions
        title = "Most Confidently Misclassified Images"
    
    indices = np.where(mask)[0]
    unique_classes = np.unique(targets)
    n_classes = len(unique_classes)
    
    # Calculate grid dimensions
    n_cols = k
    n_rows = n_classes
    
    # Create figure with extra space for title and class names
    fig = plt.figure(figsize=(3 * n_cols + 2, 3 * n_rows + 1))
    
    # Create GridSpec with extra space at top for title
    gs = plt.GridSpec(n_rows + 1, n_cols + 1, height_ratios=[0.3] + [1] * n_rows, 
                      width_ratios=[0.4] + [1] * n_cols)
    
    # Add title in the extra row at top
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(0.5, 0.5, title, fontsize=16, horizontalalignment='center', verticalalignment='center')
    ax_title.axis('off')
    
    for i, class_idx in enumerate(unique_classes):
        # Add class name or true class name in a separate column (offset by 1 row due to title)
        ax_name = fig.add_subplot(gs[i + 1, 0])
        label_text = classnames[class_idx] if mode == "correct" else f"True: {classnames[class_idx]}"
        ax_name.text(0.5, 0.5, label_text, fontsize=10, horizontalalignment='center',
                     verticalalignment='center', wrap=True)
        ax_name.axis('off')
        
        # Get indices of relevant samples for this class
        class_indices = indices[targets[indices] == class_idx]
        if len(class_indices) == 0:
            continue
        
        # Sort by similarity
        if mode == "correct":
            class_similarities = similarities[class_indices, class_idx]
        else:
            class_similarities = np.array([similarities[idx, predictions[idx]] for idx in class_indices])
        
        top_k_indices = class_indices[np.argsort(class_similarities)[-k:]]
        top_k_similarities = class_similarities[np.argsort(class_similarities)[-k:]]
        
        # Plot top k images
        for j in range(k):
            if j < len(top_k_indices):
                idx = top_k_indices[-(j+1)]
                sim = top_k_similarities[-(j+1)]
                if mode == "correct":
                    subtitle = f"Similarity: {sim:.3f}"
                else:
                    pred_class = predictions[idx]
                    subtitle = f"Pred: {classnames[pred_class]}\nSim: {sim:.3f}"
                
                ax = fig.add_subplot(gs[i + 1, j + 1])  # Offset by 1 row due to title
                ax.imshow(images_display[idx])
                ax.axis('off')
                ax.set_title(subtitle, size=8)
    
    plt.tight_layout()
    
    # Save figure with extra padding at top
    filename = 'top_correct_for_class.png' if mode == "correct" else 'top_incorrect_for_class.png'
    plt.savefig('plot/'+ filename, bbox_inches='tight', dpi=300)
    plt.show()

def plot_topk_images(images, targets, predictions, similarities, classnames, k=3, mode="correct"):
    """
    Plot the top-k best (correctly classified) or worst (misclassified) images globally across all classes.
    
    Args:
        images (np.ndarray): Array of images (N, C, H, W)
        targets (np.ndarray): Array of true labels
        predictions (np.ndarray): Array of predicted labels
        similarities (np.ndarray): Array of cosine similarities
        classnames (list): List of class names
        k (int): Number of top images to consider
        mode (str): "correct" for correctly classified, "incorrect" for misclassified
    """
    assert mode in {"correct", "incorrect"}, "Mode must be 'correct' or 'incorrect'."
    
    # Convert images from (N, C, H, W) to (N, H, W, C) and normalize for display
    images_display = np.transpose(images, (0, 2, 3, 1))
    images_display = (images_display - images_display.min()) / (images_display.max() - images_display.min())
    
    if mode == "correct":
        # Get correctly classified samples
        mask = targets == predictions
        title = "Top Best Correctly Classified Images by Cosine Similarity"
        similarity_values = similarities[np.arange(len(similarities)), targets]
    else:
        # Get misclassified samples
        mask = targets != predictions
        title = "Top Worst Misclassified Images by Cosine Similarity"
        similarity_values = np.array([similarities[idx, predictions[idx]] for idx in range(len(predictions))])
    
    indices = np.where(mask)[0]
    if len(indices) == 0:
        print(f"No {'correct' if mode == 'correct' else 'incorrect'} samples to display.")
        return
    
    # Sort indices by similarity
    sorted_indices = indices[np.argsort(similarity_values[indices])]
    top_k_indices = sorted_indices[-k:]  # Top k
    top_k_indices = top_k_indices[::-1]  # Reverse for descending order
    
    # Create figure with more space for the title
    n_cols = k
    n_rows = 1
    fig = plt.figure(figsize=(3 * n_cols, 4))  # Increased height to accommodate title
    
    # Create gridspec to manage subplot layout
    gs = plt.GridSpec(2, 1, height_ratios=[1, 8])
    
    # Add title in its own subplot
    title_ax = fig.add_subplot(gs[0])
    title_ax.axis('off')
    title_ax.text(0.5, 0.5, title, fontsize=16, ha='center', va='center')
    
    # Create subplot for images
    image_grid = gs[1].subgridspec(n_rows, n_cols)
    axes = [fig.add_subplot(image_grid[0, i]) for i in range(n_cols)]
    
    for i, idx in enumerate(top_k_indices):
        sim = similarity_values[idx]
        label_text = (f"Class: {classnames[targets[idx]]}\n"
                     f"Pred: {classnames[predictions[idx]] if mode == 'incorrect' else 'Correct'}\n"
                     f"Sim: {sim:.3f}")
        
        axes[i].imshow(images_display[idx])
        axes[i].axis('off')
        axes[i].set_title(label_text, fontsize=8, pad=5)  # Added pad for spacing
    
    plt.tight_layout()
    
    # Save figure
    filename = 'top_correct.png' if mode == "correct" else 'top_incorrect.png'
    plt.savefig('plot/'+ filename, bbox_inches='tight', dpi=300)
    plt.show()

