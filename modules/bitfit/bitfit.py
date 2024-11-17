import torch
import torch.nn.functional as F

def apply_BitFit (args, clip_model) :
    biases_list = []
    # Turn on gradients for all biases of in clip_model
    for n, p in clip_model.named_parameters():
        if 'bias' in n:
            p.requires_grad = True
            biases_list.append(p)
    
    return biases_list
    