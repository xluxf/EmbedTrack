import numpy as np
from skimage import measure


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

from matplotlib import pyplot as plt


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

    height, width = offset.shape[1], offset.shape[2]

    yxm = get_gradient(height, width)
    spatial_emb = (yxm - offset * scale) * 255

    # output structure
    waves_list = []

    # iterate over instances
    for reg_curr in measure.regionprops(instances):
        
        # mask offset by a region
        cc, rr = np.split(reg_curr.coords, 2, axis=1)
        x = spatial_emb[0, cc, rr].flatten()
        y = spatial_emb[1, cc, rr].flatten()



        if reg_curr.label == 1 and False:
            plt.scatter(x, y)

            ox = offset[0, cc, rr].flatten()
            oy = offset[1, cc, rr].flatten()

            #plt.scatter(ox, oy)

            plt.title(f'{ox.mean():.04f}, {ox.std():.04f}, {oy.mean():.04f}, {oy.std():.04f}')
            #plt.show()

        cov = np.cov(np.stack([x, y], axis=0))
        assert cov.shape == (2, 2)

        sample = [time, reg_curr.label, np.mean(x), np.mean(y), np.std(x), np.std(y), cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]]
        waves_list.append(sample)

    return waves_list
