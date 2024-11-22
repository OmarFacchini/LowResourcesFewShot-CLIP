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


def plot_confusion_matrix(targets, predictions):
    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")
    plt.show()
    
    # save plot 
    plt.savefig('confusion_matrix.png')
    
def denormalize_image(image, mean, std):
    """
    Undo normalization for visualization.
    Args:
        image (numpy.ndarray): Normalized image (C, H, W).
        mean (list or tuple): Mean values used for normalization.
        std (list or tuple): Std values used for normalization.
    Returns:
        numpy.ndarray: Denormalized image (H, W, C) in [0, 1].
    """
    image = image.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    image = image * std + mean        # Denormalize
    image = np.clip(image, 0, 1)      # Clip to valid range
    return image

def plot_evaluation_results(images, targets, predictions, similarities, k=5):
    # Identify correct and incorrect examples
    targets = np.array(targets)
    predictions = np.array(predictions)
    similarities = np.array(similarities)
    correct_mask = targets == predictions
    incorrect_mask = ~correct_mask

    correct_similarities = similarities[correct_mask, targets[correct_mask]]
    incorrect_similarities = similarities[incorrect_mask, targets[incorrect_mask]]

    # Get indices for top k correctly and incorrectly classified
    topk_correct_idx = np.argsort(correct_similarities)[-k:][::-1]
    topk_incorrect_idx = np.argsort(incorrect_similarities)[:k]
    
    # get 10 random indices
    random_idx = np.random.choice(len(images), k)
    # visualize random examples
    fig, axes = plt.subplots(1, k, figsize=(15, 3))
    for i, idx in enumerate(random_idx):
        axes[i].imshow(images[idx].transpose(1, 2, 0))
        axes[i].set_title(f"Target: {targets[idx]}, Pred: {predictions[idx]}")
        axes[i].axis('off')
    plt.suptitle(f"Random Examples")
    plt.show()
    # save
    plt.savefig(f'random_examples.png')
    
    exit()

    # Visualize top k correctly classified examples
    fig, axes = plt.subplots(1, k, figsize=(15, 3))
    for i, idx in enumerate(topk_correct_idx):
        axes[i].imshow(images[correct_mask][idx].transpose(1, 2, 0))
        axes[i].set_title(f"Sim: {correct_similarities[idx]:.2f}")
        axes[i].axis('off')
    plt.suptitle(f"Top {k} Correctly Classified")
    plt.show()
    # save
    plt.savefig(f'top{k}_correct.png')

    # Visualize top k incorrectly classified examples
    fig, axes = plt.subplots(1, k, figsize=(15, 3))
    for i, idx in enumerate(topk_incorrect_idx):
        axes[i].imshow(images[incorrect_mask][idx].transpose(1, 2, 0))
        axes[i].set_title(f"Sim: {incorrect_similarities[idx]:.2f}")
        axes[i].axis('off')
    plt.suptitle(f"Top {k} Incorrectly Classified")
    plt.show()
    # save
    plt.savefig(f'top{k}_incorrect.png')
