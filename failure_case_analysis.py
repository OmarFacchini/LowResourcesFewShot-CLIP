import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skimage.transform import resize
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

METRICS = False
UMAP_PLOT = True
FAILURES_PLOT = False
DATASET = 'eurosat'

"""
GET DATA FUNCTIONS
"""

def get_data(csv_filename='results/results_csv/zero_shot_eurosat.csv', json_label_map='data/eurosat/label_map.json'):
    # Open and read the JSON file
    with open(json_label_map, 'r') as file:
        label_map = json.load(file)

    # Load the CSV file using pandas
    df = pd.read_csv(csv_filename)
    # Extract features and targets from the dataframe
    features = np.array(df['features'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist())  # Convert string to list of numbers
    similarities = np.array(df['similarity'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist())  # Convert string to list of numbers
    targets = np.array(df['target'])
    predictions = np.array(df['prediction'])
    
    # Map numeric labels to their corresponding string labels
    string_targets = np.array([label_map[str(target)] for target in targets])
    
    return features, targets, predictions, similarities, string_targets, label_map

"""
METRICS COMPUTATION FUNCTIONS
"""

def compute_silhouette_scores(embeddings, targets, predictions):
    # Silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
    silhouette_complete = silhouette_score(embeddings, targets, metric='euclidean')
    correct_indices = targets == predictions
    silhouette_correct = silhouette_score(embeddings[correct_indices], targets[correct_indices], metric='euclidean') if np.sum(correct_indices) > 0 else None
    wrong_indices = ~correct_indices
    silhouette_wrong = silhouette_score(embeddings[wrong_indices], targets[wrong_indices], metric='euclidean') if np.sum(wrong_indices) > 0 else None

    return silhouette_complete, silhouette_correct, silhouette_wrong

def compute_ari(true_labels, predicted_labels):
    # ARI is a measure of the similarity between cluster assignments.
    # It's robust to cluster imbalance.
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari

def compute_clustering_metrics(true_labels, predicted_labels):
    # Measures whether each cluster contains only data points that are members of a single ground truth class.
    # It's a synonym of cluster "purity".
    # "Are the clusters pure with respect to the ground truth?"
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    # Ensures that all data points from a single ground truth class are assigned to the same predicted cluster.
    # "Are all points in a ground truth class assigned to the same cluster?"
    completeness = completeness_score(true_labels, predicted_labels)
    # Harmonic mean of homogeneity and completeness.
    v_measure = v_measure_score(true_labels, predicted_labels)
    return homogeneity, completeness, v_measure

def compute_class_accuracy(targets, predictions, string_targets):
    class_metrics = []
    unique_classes = np.unique(targets)
    
    for cls in unique_classes:
        class_indices = targets == cls
        total_class_count = np.sum(class_indices)
        
        correct_class_count = np.sum(class_indices & (targets == predictions))  # Correct predictions
        wrong_class_count = total_class_count - correct_class_count  # Wrong predictions
        
        accuracy = correct_class_count / total_class_count if total_class_count > 0 else 0
        
        class_name = string_targets[class_indices][0]  # Assuming all class indices map to the same string name
        
        class_metrics.append([class_name, correct_class_count, wrong_class_count, total_class_count, accuracy])

    class_accuracy_df = pd.DataFrame(class_metrics, columns=["Class", "+", "-", "Total", "Accuracy"])
    return class_accuracy_df

"""
PLOTTING FUNCTIONS
"""

def plot_umap(features, targets, predictions, string_targets, output_filename='results/plot/umap_plot.png'):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_embeddings = umap_model.fit_transform(features)

    x_min, x_max = umap_embeddings[:, 0].min() - 0.1, umap_embeddings[:, 0].max() + 0.1
    y_min, y_max = umap_embeddings[:, 1].min() - 0.1, umap_embeddings[:, 1].max() + 0.1

    unique_targets = np.unique(string_targets)
    vibrant_colors = sns.color_palette("hls", n_colors=len(unique_targets))
    color_map = {target: vibrant_colors[i] for i, target in enumerate(unique_targets)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                       c=[color_map[string_targets[i]] for i in range(len(targets))], s=12)
    axes[0, 0].set_title('Complete Set', fontsize=14)
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    axes[0, 0].set_xlim(x_min, x_max)
    axes[0, 0].set_ylim(y_min, y_max)

    correct_indices = targets == predictions
    point_colors = []
    for i in range(len(targets)):
        if correct_indices[i]:
            color = np.array(color_map[string_targets[i]])
            color = np.concatenate((color, [1.0]))  # Full opacity
        else:
            color = np.array(color_map[string_targets[i]])
            color = np.concatenate((color, [0.3]))  # Add alpha for transparency
        point_colors.append(color)

    axes[0, 1].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                       c=point_colors, s=12)
    axes[0, 1].set_title('Merged Prediction (Wrongs in Transparent)', fontsize=14)
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    axes[0, 1].set_xlim(x_min, x_max)
    axes[0, 1].set_ylim(y_min, y_max)

    axes[1, 0].scatter(umap_embeddings[correct_indices, 0], umap_embeddings[correct_indices, 1], 
                       c=[color_map[string_targets[i]] for i in range(len(targets)) if correct_indices[i]], s=12)
    axes[1, 0].set_title('Correct Predictions', fontsize=14)
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_ylim(y_min, y_max)

    wrong_indices = ~correct_indices
    axes[1, 1].scatter(umap_embeddings[wrong_indices, 0], umap_embeddings[wrong_indices, 1], 
                       c=[color_map[string_targets[i]] for i in range(len(targets)) if wrong_indices[i]], s=12)
    axes[1, 1].set_title('Wrong Predictions', fontsize=14)
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(y_min, y_max)

    plt.tight_layout()

    # Adjust the legend to have two rows
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in vibrant_colors]
    labels = [str(target) for target in unique_targets]
    fig.legend(handles, labels, loc='lower center', ncol=len(unique_targets)//5, bbox_to_anchor=(0.5, -0.15), fontsize=12)

    # Save the combined image
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Combined UMAP plots saved to {output_filename}\n")
    

def plot_confusion_matrix(targets, predictions, classnames):
    '''
    Plot confusion matrix with improved handling of long class names
    
    Args:
        targets: True labels
        predictions: Predicted labels
        classnames: List of class names
    '''
    classnames = [classnames[str(target)] for target in (set(targets).union(set(predictions)))]
    
    #print(classnames)
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
    
    if DATASET == "eurosat":
        # Adjust label properties
        plt.xticks(fontsize=8, ha='right')  # Align rotated labels to the right
        plt.yticks(fontsize=8)
    else:
        plt.xticks(fontsize=6, ha='right')  # Align rotated labels to the right
        plt.yticks(fontsize=6)
        for text in disp.text_.ravel():
            text.set_fontsize(4)  # Adjust internal font size for digits
    
    # Add title with padding
    plt.title('Confusion Matrix', pad=20)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure with high DPI
    plt.savefig('results/plot/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_topk_images_for_class(
    images, targets, predictions, similarities, classnames, k=3, model=None, preprocess=None, dataset=None, mode="correct"
):
    """
    Plot top-k images for each class with their attention maps based on correctness and similarity, in a compact layout.

    Args:
        images (np.ndarray): Array of images (N, C, H, W)
        targets (np.ndarray): Array of true labels
        predictions (np.ndarray): Array of predicted labels
        similarities (np.ndarray): Array of cosine similarities
        classnames (list): List of class names
        k (int): Number of top images to consider per class
        model: Model used for generating attention maps
        preprocess: Preprocessing function for the model
        dataset: Dataset object to access image paths
        mode (str): "correct" for correctly classified, "incorrect" for misclassified
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from skimage.transform import resize

    assert mode in {"correct", "incorrect"}, "Mode must be 'correct' or 'incorrect'."

    # Convert images from (N, C, H, W) to (N, H, W, C) and normalize for display
    images_display = np.transpose(images, (0, 2, 3, 1))
    images_display = (images_display - images_display.min()) / (images_display.max() - images_display.min())

    # Determine mask and title based on mode
    if mode == "correct":
        mask = targets == predictions
        title = "Top Correctly Classified Images with Attention Maps"
    else:
        mask = targets != predictions
        title = "Most Confidently Misclassified Images with Attention Maps"

    indices = np.where(mask)[0]
    unique_classes = np.unique(targets)
    n_classes = len(unique_classes)

    # Calculate grid dimensions: 2 rows per class (images and attention maps), + 1 for separators
    n_rows = 2 * n_classes + (n_classes - 1)  # Includes separator rows
    n_cols = k + 1  # Extra column for class names

    # Enable constrained layout
    fig = plt.figure(figsize=(2 * n_cols, 1.5 * n_rows), constrained_layout=True)
    gs = GridSpec(n_rows, n_cols, figure=fig)

    current_row = 0
    for i, class_idx in enumerate(unique_classes):
        # Add class name in the first column (row index)
        ax_label = fig.add_subplot(gs[current_row:current_row + 2, 0])
        ax_label.axis('off')
        ax_label.text(
            0.5, 0.5, classnames[class_idx], fontsize=10, ha='center', va='center', wrap=True
        )

        # Get indices of relevant samples for this class
        class_indices = indices[targets[indices] == class_idx]
        if len(class_indices) == 0:
            current_row += 3  # Skip a row for separator
            continue

        # Sort by similarity
        if mode == "correct":
            class_similarities = similarities[class_indices, class_idx]
        else:
            class_similarities = np.array([similarities[idx, predictions[idx]] for idx in class_indices])

        top_k_indices = class_indices[np.argsort(class_similarities)[-k:]]
        top_k_similarities = class_similarities[np.argsort(class_similarities)[-k:]]

        # Plot top k images and their attention maps
        for j in range(k):
            if j < len(top_k_indices):
                idx = top_k_indices[-(j+1)]
                sim = top_k_similarities[-(j+1)]
                if mode == "correct":
                    subtitle = f"Sim: {sim:.3f}"
                else:
                    pred_class = predictions[idx]
                    subtitle = f"Pred: {classnames[pred_class]}\nSim: {sim:.3f}"

                # Plot original image
                ax_img = fig.add_subplot(gs[current_row, j + 1])
                ax_img.imshow(images_display[idx])
                ax_img.axis('off')
                ax_img.set_title(subtitle, size=8, pad=2)

                # Plot attention map
                attention, salient = plot_attention_map_enhance(dataset[idx].impath, preprocess, model, idx, False)
                h, w = images_display[idx].shape[:2]
                salient_heatmap = np.zeros_like(attention)
                salient_heatmap[salient] = attention[salient]
                salient_heatmap_resized = resize(salient_heatmap, (h, w), order=3, mode='constant')

                ax_attn = fig.add_subplot(gs[current_row + 1, j + 1])
                ax_attn.imshow(images_display[idx])
                ax_attn.imshow(salient_heatmap_resized, alpha=0.3, cmap='jet')
                ax_attn.axis('off')

        # Move to next set of rows, with a separator
        current_row += 3

    # Adjust layout
    fig.suptitle(title, fontsize=14)
    fig.savefig(f"top_{mode}_images_for_class.png", dpi=300, bbox_inches='tight')
    plt.show()


  


def plot_topk_images(images, targets, predictions, similarities, classnames, k=3, model=None, preprocess=None, dataset=None, mode="correct"):
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
        model: Model used for generating attention maps
        preprocess: Preprocessing function for the model
        dataset: Dataset object to access image paths
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize

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

    # Create figure with space for original images and attention maps
    n_cols = k
    n_rows = 2  # One row for original images, one for attention maps
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8))

    for i, idx in enumerate(top_k_indices):
        sim = similarity_values[idx]
        label_text = (f"Class: {classnames[targets[idx]]}\n"
                      f"Pred: {classnames[predictions[idx]] if mode == 'incorrect' else 'Correct'}\n"
                      f"Sim: {sim:.3f}")
        
        # Compute attention map
        attention, salient = plot_attention_map_enhance(dataset[idx].impath, preprocess, model, idx, False)

        # Original image
        axes[0, i].imshow(images_display[idx])
        axes[0, i].axis('off')
        axes[0, i].set_title(label_text, fontsize=8, pad=5)

        # Attention map
        h, w = images_display[idx].shape[:2]
        salient_heatmap = np.zeros_like(attention)
        salient_heatmap[salient] = attention[salient]
        salient_heatmap_resized = resize(salient_heatmap, (h, w), order=3, mode='constant')

        axes[1, i].imshow(images_display[idx])
        axes[1, i].imshow(salient_heatmap_resized, alpha=0.3, cmap='jet')
        axes[1, i].axis('off')
        axes[1, i].set_title("Attention Map", fontsize=8, pad=5)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results/top_{mode}_images.png", dpi=300, bbox_inches='tight')
    # plt.show()


def plot_attention_map(impath, preprocess, clip_model, name):
    transform_image = transforms.Compose([
        transforms.Resize(clip_model.visual.input_resolution, interpolation=Image.BICUBIC),
        transforms.CenterCrop(clip_model.visual.input_resolution),
        lambda image: image.convert("RGB"),
    ])

    img = Image.open(impath) # PIL img
    img_input = preprocess(img).unsqueeze(0)

    with torch.no_grad(): 
        image_attention = clip_model.encode_image_attention(img_input)
        #reshaped_attention = image_attention[0].reshape(14, 14)

    #original_img = transform_image(img)
    #overlayed_img = overlay_transparency(original_img, reshaped_attention)

    fig = plt.figure(figsize=[10, 5], frameon=False)
    ax = fig.add_subplot(1, 2, 1)
    ax.axis("off")
    ax.imshow(transform_image(img))
    ax = fig.add_subplot(1, 2, 2)
    ax.axis("off")
    ax.imshow(image_attention[0].reshape(14, 14))
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(f"{name}_{1}")


def plot_attention_map_enhance(impath, preprocess, model, name, plot=True):
    # Transformation to match CLIP model's input requirements
    transform_image = transforms.Compose([
        transforms.Resize(model.visual.input_resolution, interpolation=Image.BICUBIC),
        transforms.CenterCrop(model.visual.input_resolution),
        lambda image: image.convert("RGB"),
    ])

    # Open and preprocess the image
    img = Image.open(impath)
    img_input = preprocess(img).unsqueeze(0).cuda()

    # Get attention map
    with torch.no_grad(): 
        image_attention = model.encode_image_attention(img_input)
        attention_map = image_attention[0].reshape(14, 14).cpu().numpy()

    # Normalize attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Create a heatmap with a threshold to highlight most salient regions
    threshold = np.percentile(attention_map, 50)  # Adjust this percentile as needed
    salient_mask = attention_map >= threshold
    
    if plot:
    # Create figure
        fig = plt.figure(figsize=[10, 5])
        ax = fig.add_subplot(1, 2, 1)
        
        # Original image
        original_img = transform_image(img)
        ax.imshow(original_img)
        ax.set_title('Original Image')
        ax.axis('off')
        
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('Attention Map Overlay')
        ax.imshow(original_img)

        # Overlay salient attention map
        cmap = plt.cm.get_cmap('jet')  # You can change 'jet' to other colormaps like 'viridis', 'plasma', etc.
        salient_heatmap = np.zeros_like(attention_map)
        salient_heatmap[salient_mask] = attention_map[salient_mask]
        
        # Resize attention map to match image dimensions
        
        salient_heatmap_resized = resize(salient_heatmap, (original_img.height, original_img.width), 
                                        order=3, mode='constant')
        
        # Color the most salient regions
        ax.imshow(cmap(salient_heatmap_resized), alpha=0.3, cmap=cmap)
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{name}_attention.png", bbox_inches='tight', pad_inches=0)
        plt.close()
    return attention_map, salient_mask


if __name__ == "__main__":
    # Load data
    features, targets, predictions, similarities, string_targets, classnames = get_data()

    if METRICS:
        with open("metrics.txt", "w") as myfile:
            silhouette_complete, silhouette_correct, silhouette_wrongs = compute_silhouette_scores(features, targets, predictions)
            print(f"Silhouette Score : T {silhouette_complete:.4f}, C {silhouette_correct:.4f}, W {silhouette_wrongs:.4f}")
            myfile.write(f"Silhouette Score : T {silhouette_complete:.4f}, C {silhouette_correct:.4f}, W {silhouette_wrongs:.4f}\n")

            ari = compute_ari(targets, predictions)
            print(f"ARI : {ari:.4f}")
            myfile.write(f"ARI : {ari:.4f}\n")

            homogeneity, completeness, v_measure = compute_clustering_metrics(targets, predictions)
            print(f"Homogeneity : {homogeneity:.4f}, Completeness : {completeness:.4f}, V-measure : {v_measure:.4f}")
            myfile.write(f"Homogeneity : {homogeneity:.4f}, Completeness : {completeness:.4f}, V-measure : {v_measure:.4f}\n")


            class_accuracy_df = compute_class_accuracy(targets, predictions, string_targets)
            print("\nClass-wise Accuracy:")
            print(class_accuracy_df)
            myfile.write(f"Class-wise Accuracy: {class_accuracy_df}")


    if UMAP_PLOT:
        plot_umap(features, targets, predictions, string_targets)

    if FAILURES_PLOT :
        plot_confusion_matrix(targets, predictions, classnames)

