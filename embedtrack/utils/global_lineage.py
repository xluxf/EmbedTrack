"""
Author: Filip Lux (2023), Karlsruhe Institute of Technology
Licensed under MIT License
"""

import numpy as np


def get_gradient(height, width):
    """

    :param height:
    :param width:
    :return:
    """

    # TODO: save yxm as a global structure

    # create gradient images
    grad_x = np.expand_dims(np.linspace(0, height, height, dtype=float, endpoint=False) / 255, axis=-1)
    grad_x = np.repeat(grad_x, width, axis=1)

    grad_y = np.expand_dims(np.linspace(0, width, width, dtype=float, endpoint=False) / 255, axis=0)
    grad_y = np.repeat(grad_y, height, axis=0)

    yxm = np.stack((grad_x, grad_y), 0)
    return yxm


def get_offset_wavefunc(instances, offset, time, scale=1):
    """
    Get probabilities of edges.
    smooth version
    Args:
        instances: torch.tensor
            labeled image instances in the time 't'
        offset: torch.tensor
            the tracking offset, two channels
        time:
            frame index

    Returns:
        edge_list : list
            list of tuples describing edges in a format (curr_time, prev_time, label_curr, label_prev, prob)
    """
    
    # TODO: add TANH activation if offsets

    height, width = offset.shape[1], offset.shape[2]

    yxm = get_gradient(height, width)
    spatial_emb = (yxm - offset[:2] * scale) * 255

    # output structure
    waves_list = []

    # iterate over instances
    for label in np.unique(instances):
        if label == 0:
            continue
        cc, rr = (instances == label).nonzero()

        # mask offset by a region
        x = spatial_emb[0, cc, rr].flatten()
        y = spatial_emb[1, cc, rr].flatten()

        sample = [time, label, np.mean(x), np.mean(y), np.std(x), np.std(y), 'c00', 'c01', 'c10', 'c11']
        waves_list.append(sample)

    return waves_list
