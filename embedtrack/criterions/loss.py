"""
Original work Copyright 2019 Davy Neven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler (MIT License)
Modifications: changed IOU calculation; extended with tracking loss
"""

import torch
import torch.nn as nn
from embedtrack.criterions.lovasz_losses import lovasz_hinge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbedTrackLoss(nn.Module):
    def __init__(
        self,
        cluster,
        grid_y=1024,
        grid_x=1024,
        pixel_y=1,
        pixel_x=1,
        n_sigma=2,
        foreground_weight=1,
    ):
        """
        Loss for training the EmbedTrack net.
        Args:
            cluster (Callable): cluster the predictions to instances
            grid_x (int): size of the grid in x direction
            grid_y (int): size of the grid in y direction
            pixel_x (int): size of a pixel
            pixel_y (int): size of a pixel
            n_sigma (int): number of channels estimating sigma (which is used to estimate the object size)
            foreground_weight (int): weight of the foreground compare to the background
        """
        super().__init__()

        print(
            "Created spatial emb loss function with: n_sigma: {}, foreground_weight: {}".format(
                n_sigma, foreground_weight
            )
        )
        print(f"grid size: {grid_x}x{grid_y}")
        print("*************************")
        self.cluster = cluster
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # coordinate map
        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        yxm = torch.cat((ym, xm), 0)

        self.register_buffer("yxm", yxm)
        self.register_buffer("yx_shape", torch.tensor(self.yxm.size()[1:]).view(-1, 1))

    def forward(
        self,
        predictions,
        instances,
        labels,
        center_images,
        offsets,
        w_inst=1,
        w_var=10,
        w_seed=1,
        w_track=2,
        iou=False,
        iou_meter=None,
    ):
        """

        Args:
            predictions (tuple): tuple of torch tensors containing the network output
            instances (torch.Tensor): ground truth instance segmentation masks
            labels (torch.Tensor): semantic segmentation masks
            center_images (torch.Tensor): masks containing the gt cell center position
            offsets (torch.Tensor): masks containing the shift between cell centers of two successive frames
            w_inst (int): weight for the instance loss
            w_var (int): weight for the variance loss
            w_seed (int): weight for the seed loss
            w_track (int):weight for the tracking loss
            iou (bool): if True, calculate the IOU of the instance segmentation
            iou_meter (Callable): contains the calculated IOU scores

        Returns: (torch.Tensor) loss, (dict) values of the different loss parts

        """

        # segmentation_prediction.shape == (b, 5, w, h)
        # tracking_predictions.shape == (b, 4, w, h)
        segmentation_predictions, tracking_predictions = predictions

        # instances B 1 Y X
        batch_size, height, width = (
            segmentation_predictions.size(0),
            segmentation_predictions.size(2),
            segmentation_predictions.size(3),
        )

        yxm_s = self.yxm[:, 0:height, 0:width].contiguous()  # N x h x w if 2D images: N=2

        # reported loss values
        loss_values = {
            "instance": torch.tensor(0.),
            "var_instance": torch.tensor(0.),
            "seed": torch.tensor(0.),
            "tracking": torch.tensor(0.),
            "var_tracking": torch.tensor(0.)
        }

        # only instances
        # similar to embedSeg
        seg_loss = torch.tensor(0., device=device, requires_grad=True)
        tra_loss = torch.tensor(0., device=device, requires_grad=True)

        for b in range(0, batch_size):

            # count of objects
            obj_count = 0

            # process segmentation prediction
            spatial_emb = torch.tanh(segmentation_predictions[b, 0:2]) + yxm_s        # 2 x h x w
            # TODO: INFO sigma is originaly not activated in EmbedSeg
            sigma_seg = torch.sigmoid(segmentation_predictions[b, 2:2+self.n_sigma])  # n_sigma x h x w
            seed_map = torch.sigmoid(segmentation_predictions[b, 2+self.n_sigma:2+self.n_sigma+1])  # 1 x h x w

            # loss accumulators
            var_inst_loss = 0
            instance_loss = 0
            seed_loss = 0

            instance = instances[b].unsqueeze(0)    # 1 x h x w
            label = labels[b].unsqueeze(0)          # 1 x h x w
            center_image = center_images[b].unsqueeze(0)  # 1 x h x w

            # get instances
            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress SEED bg to zero
            bg_mask = label == 0

            if bg_mask.sum() > 0:
                seed_loss = seed_loss + torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))

            # TODO: train also background image
            if len(instance_ids) == 0:  # background image
                continue

            for inst_id in instance_ids:

                in_mask = instance.eq(inst_id)                                        # mask of the object, 1 x h x w
                center_mask = in_mask & center_image                                  # location of the seg center

                # TODO: resolve why there is no overlap
                if center_mask.sum().eq(0):                                           # no object
                    #print('SEG: no overlap of in_mask and center_image')
                    continue
                if center_mask.sum().eq(1):
                    center = yxm_s[center_mask.expand_as(yxm_s)].view(2, 1, 1)        # indexes of the center
                else:
                    print('SEG: multiple overlaps of in_mask and center_image')
                    xy_in = yxm_s[in_mask.expand_as(yxm_s)].view(
                        2, -1
                    )  # TODO --> should this edge case change!
                    center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # calculate sigma
                sigma_in = sigma_seg[in_mask.expand_as(sigma_seg)].view(self.n_sigma, -1)
                s_seg = sigma_in.mean(dim=1).view(self.n_sigma, 1, 1)  # n_sigma x 1 x 1

                # calculate var loss before exp
                var_inst_loss = var_inst_loss + torch.mean(torch.pow(sigma_in - s_seg.detach(), 2))

                # if sigmoid constrained 0...1 before exp afterwards scale 1...22026 - more than enough range to
                # simulate pix size objects and large objects!
                s_seg = torch.exp(
                    s_seg * 10
                )

                dist = torch.exp(
                    - torch.sum(torch.pow(spatial_emb - center, 2) * s_seg,
                                dim=0,
                                keepdim=True)
                )

                # apply lovasz-hinge loss
                instance_loss = instance_loss + lovasz_hinge(
                    dist * 2 - 1, in_mask.to(device)
                )

                # seed loss
                seed_loss += self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2)
                )

            # validation
            # calculate instance IOU
            if iou:
                instance_pred = self.cluster.cluster_pixels(
                    segmentation_predictions[b], n_sigma=2, return_on_cpu=False
                )
                iou_scores = calc_iou_full_image(
                    instances[b].detach(),
                    instance_pred.detach(),
                )
                for score in iou_scores:
                    iou_meter.update(score)

                # TODO: do the same for tracking predictions

            # normalize losses
            if obj_count > 0:
                instance_loss /= obj_count
                var_inst_loss /= obj_count
            seed_loss = seed_loss / (height * width)

            seg_loss = seg_loss +\
                       w_inst * instance_loss +\
                       w_var * var_inst_loss +\
                       w_seed * seed_loss

            loss_values["instance"] += w_inst * (
                instance_loss.detach().cpu()
                if isinstance(instance_loss, torch.Tensor) else torch.tensor(instance_loss,
                                                                             dtype=torch.float,
                                                                             device='cpu')
            )
            loss_values["var_instance"] += w_var * (
                var_inst_loss.detach().cpu()
                if isinstance(var_inst_loss, torch.Tensor) else torch.tensor(var_inst_loss,
                                                                             dtype=torch.float,
                                                                             device='cpu')
            )
            loss_values["seed"] += w_seed * (
                seed_loss.detach().cpu()
                if isinstance(seed_loss, torch.Tensor) else torch.tensor(seed_loss,
                                                                         dtype=torch.float,
                                                                         device='cpu')
            )

        # TRACKING
        # segmentation branch predictions where concatenated (frames t, frames t-1)
        # since tracking offset is calculated between t->t-1
        # the tracking batch has only half the length compared to the segmentation predictions
        half_batch_size = batch_size // 2
        for b in range(half_batch_size):

            tra_obj_count = 0
            tracking_loss = 0
            var_tracking_loss = 0

            tracking_emb = yxm_s - torch.tanh(tracking_predictions[b, 0:2])
            # TODO: not activated in EmbedSeg
            sigma_tra = torch.sigmoid(tracking_predictions[b, 2:2 + self.n_sigma])  # n_sigma x h x w

            instance_curr = instances[b].unsqueeze(0)    # 1 x h x w
            center_image_curr = center_images[b].unsqueeze(0)  # 1 x h x w
            offset = offsets[b]

            instance_ids = instance_curr.unique()
            instance_ids = instance_ids[instance_ids != 0]

            for inst_id in instance_ids:

                in_mask = instance_curr.eq(inst_id)
                center_mask = in_mask & center_image_curr
                center = yxm_s[center_mask.expand_as(yxm_s)].view(-1, 1, 1)

                # an instance needs to have the center in a prev frame
                if center_mask.sum() == 0:
                    continue

                assert center_mask.sum() == 1, center_mask.sum()

                # calculate a proper offset
                # TODO: replace 256 with self.grid_x
                gt_tra_center = (center - offset[center_mask.expand_as(offset)].view(-1, 1, 1) / 256).view(-1, 1, 1).float()
                if (gt_tra_center.min() < 0) or (gt_tra_center.max() > 1):
                    # print(f'ERR: excluding gt_tra_center {list(gt_tra_center.detach().cpu().numpy())}')
                    continue

                # assert len(gt_tra_center) == 2, gt_tra_center
                # assert gt_tra_center.shape == (2, 1, 1), gt_tra_center.shape

                # calculate sigma
                sigma_in_tra = sigma_tra[in_mask.expand_as(sigma_tra)].view(self.n_sigma, -1)
                s_tra = sigma_in_tra.mean(1).view(self.n_sigma, 1, 1)  # n_sigma x 1 x 1

                # assert len(s_tra) == 2, s_tra
                # assert s_tra.shape == (2, 1, 1), s_tra.shape

                # calculate var loss before exp
                var_tracking_loss = var_tracking_loss + torch.sum(torch.pow(sigma_in_tra - s_tra.detach(), 2))

                s_tra = torch.exp(
                    s_tra * 10
                )

                dist_tracking = torch.exp(
                    - torch.sum(
                        torch.pow(tracking_emb - gt_tra_center, 2) * s_tra,
                        0,
                        keepdim=True,
                    )
                )
                tracking_loss = tracking_loss + lovasz_hinge(
                    dist_tracking * 2 - 1, in_mask.to(device)
                )
                tra_obj_count += 1

            # normalize losses
            if tra_obj_count > 0:
                tracking_loss /= tra_obj_count
                var_tracking_loss /= tra_obj_count

            tra_loss = tra_loss +\
                       w_inst * tracking_loss +\
                       w_var * var_tracking_loss

            loss_values["tracking"] += 2 * w_inst * (
                tracking_loss.detach().cpu()
                if isinstance(tracking_loss, torch.Tensor) else torch.tensor(tracking_loss,
                                                                             dtype=torch.float,
                                                                             device='cpu')
            )
            loss_values["var_tracking"] += 2 * w_var * (
                var_tracking_loss.detach().cpu()
                if isinstance(var_tracking_loss, torch.Tensor) else torch.tensor(var_tracking_loss,
                                                                                 dtype=torch.float,
                                                                                 device='cpu')
            )

        # there are only half of tra samples
        loss = seg_loss + 2 * tra_loss
        loss = loss / batch_size

        for key in loss_values.keys():
            loss_values[key] = loss_values[key] / batch_size

        return loss, loss_values

def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou


def calc_iou_full_image(gt, prediction):
    """Calculate all IOUs in the image crop"""
    gt_labels = torch.unique(gt)
    gt_labels = gt_labels[gt_labels > 0]
    pred_labels = prediction[prediction > 0].unique()
    # go through gt labels
    ious = []
    matched_pred_labels = []
    for gt_l in gt_labels:
        gt_mask = gt.eq(gt_l)
        overlapping_pred_labels = prediction[gt_mask].unique()
        overlapping_pred_labels = overlapping_pred_labels[overlapping_pred_labels > 0]
        if not len(overlapping_pred_labels):  # false negative
            ious.append(0)
            continue
        # otherwise assign to gt mask the prediction with largest iou
        # calculate_iou returns single float which is on the cpu
        gt_ious = torch.tensor(
            [
                calculate_iou(gt_mask, prediction.eq(p_l))
                for p_l in overlapping_pred_labels
            ]
        )
        if len(gt_ious) > 0:
            idx_max_iou = torch.argmax(gt_ious)
            ious.append(gt_ious[idx_max_iou])
            matched_pred_labels.append(overlapping_pred_labels[idx_max_iou])

    # add not matched pred labels by adding iou==0 (FPs)
    if len(matched_pred_labels) > 0:
        matched_pred_labels = torch.stack(matched_pred_labels)
        # (pred_labels[..., None] == matched_pred_labels).any(-1) equvalent to np.isin(pred_labels, matched_pred_labels)
        num_non_matched = (
            ~(pred_labels[..., None] == matched_pred_labels).any(-1)
        ).sum()
    else:
        num_non_matched = len(pred_labels)
    ious.extend([0] * num_non_matched)
    return ious
