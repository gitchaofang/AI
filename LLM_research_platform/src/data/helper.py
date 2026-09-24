import regex as re
import torch
import numpy as np
from PIL import Image
import totch.nn.functional as F


# text tokenization
def get_stats(ids, counts=None): #update counts
    counts = {} if counts is None else counts
    for pair in zip(ids[:-1],ids[1:]):
        counts[pair] = counts.get(pair,0) + 1

def merge(ids,pair,idx):
    newids = []
    i = 0
    while i < len(ids):
        if (ids[i] == pair[0] and i != len(ids) - 1 and ids[i + 1] == pair[1]):
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# image processing
def imageToTensor(image):
    # PIL image -> [C,H,W] float tensor in [0,1]
    array = np.array(image)
    tensor = torch.from_np(array)
    tensor = tensor.permute(2,0,1)
    tensor = tensor.float() / 225.0

    return tensor

def pad_to_patch_grid(image, patch_size = 16, pad_value = [0.0,0.0,0.0,]): 
    '''
    image: [C,H,W]
    pad_value: [C]
    '''
    C, H ,W = image.shape
    H_pad = ((H + patch_size - 1) // patch_size) * patch_size
    W_pad = ((H + patch_size - 1) // patch_size) * patch_size 

    padded = image.new_empty(C,H_pad, W_pad)

    # Fill with channel-specific values
    for c in range(C):
        padded[C].fill_(pad_value[c])

    #Copy original image
    padded[:,:H,:W] = image
    return padded

def make_2d_positions(H_patches, W_patches):
    """
    Returns:
        positions: [N, 2]

    positions[:, 0] = x / column
    positions[:, 1] = y / row
    """

    y, x = torch.meshgrid(
        torch.arange(H_patches),
        torch.arange(W_patches),
        indexing="ij",
    )

    positions = torch.stack(
        [x.flatten(), y.flatten()],
        dim=-1,
    )

    return positions

def patchify(image, patch_size = 16, pad_value = [0.0, 0.0, 0.0]):
    '''
    image: [C,H,W]
    returns:
        patches: [N, C * patch_size * patch_size]
    '''
    # make images ready for patchify
    pad_image = pad_to_patch_grid(image, patch_size, pad_value = pad_value)

    C,H,W = image.shape
    assert H % patch_size == 0
    assert W % patch_size == 0

    # [C,W,H]

    patches = pad_image.unfold(dimension=1, size=patch_size, step=patch_size)
    patches = patches.unfold(dimension=2, size=patch_size, step=patch_size)

    # [C, H_patches, W_patches, P, P]
    patches = patches.permute(1,2,0,3,4)

    H_patches = H // patch_size
    W_patches = W // patch_size

    positions = make_2d_positions(H_patches=H_patches, W_patches=W_patches)

    # [N, C * P * P]
    patches = patches.reshape(
        H_patches * W_patches,
        C * patch_size * patch_size,
    )

    return {"patches": patches,
            "positions": positions}



