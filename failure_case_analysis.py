import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

METRICS = True
UMAP_PLOT = True
DATASET = 'eurosat'

def get_data(csv_filename='data/evaluation_results.csv', json_label_map='data/eurosat/label_map.json'):
    # Open and read the JSON file
    with open(json_label_map, 'r') as file:
        label_map = json.load(file)

    # Load the CSV file using pandas
    df = pd.read_csv(csv_filename)
    # Extract features and targets from the dataframe
    features = np.array(df['features'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist())  # Convert string to list of numbers
    targets = np.array(df['target'])
    predictions = np.array(df['prediction'])
    
    # Map numeric labels to their corresponding string labels
    string_targets = np.array([label_map[str(target)] for target in targets])
    
    return features, targets, string_targets, predictions

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

def plot_umap(features, targets, predictions, string_targets, output_filename='plot/umap_plot.png'):
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
    fig.legend(handles, labels, loc='lower center', ncol=len(unique_targets)//2, bbox_to_anchor=(0.5, -0.1), fontsize=12)

    # Save the combined image
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Combined UMAP plots saved to {output_filename}\n")
    
if __name__ == "__main__":
    features, targets, string_targets, predictions = get_data()

    if METRICS:
        silhouette_complete, silhouette_correct, silhouette_wrongs = compute_silhouette_scores(features, targets, predictions)
        print(f"Silhouette Score : T {silhouette_complete:.4f}, C {silhouette_correct:.4f}, W {silhouette_wrongs:.4f}")
        ari = compute_ari(targets, predictions)
        print(f"ARI : {ari:.4f}")
        homogeneity, completeness, v_measure = compute_clustering_metrics(targets, predictions)
        print(f"Homogeneity : {homogeneity:.4f}, Completeness : {completeness:.4f}, V-measure : {v_measure:.4f}")

        class_accuracy_df = compute_class_accuracy(targets, predictions, string_targets)
        print("\nClass-wise Accuracy:")
        print(class_accuracy_df)

    if UMAP_PLOT:
        plot_umap(features, targets, predictions, string_targets)
