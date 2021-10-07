import paddle
import paddle.nn as nn
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, display


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = paddle.minimum(paddle.unsqueeze(a[:, 3], axis=1), b[:, 2]) - paddle.maximum(paddle.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = paddle.minimum(paddle.unsqueeze(a[:, 2], axis=1), b[:, 3]) - paddle.maximum(paddle.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = paddle.clip(iw, min=0)
    ih = paddle.clip(ih, min=0)
    ua = paddle.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = paddle.clip(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Layer):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            # bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            # bbox_annotation = bbox_annotation.masked_select((bbox_annotation[:, 4] != -1).unsqueeze(1).tile([1,bbox_annotation.shape[1]])).reshape([-1, bbox_annotation.shape[1]])
            bbox_annotation = paddle.to_tensor(bbox_annotation.numpy()[bbox_annotation[:, 4].numpy() != -1])
            classification = paddle.clip(classification, 1e-4, 1.0 - 1e-4)
            
            if bbox_annotation.shape[0] == 0:
                alpha_factor = paddle.ones_like(classification) * alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)
                
                bce = -(paddle.log(1.0 - classification))
                
                cls_loss = focal_weight * bce
                
                regression_losses.append(paddle.to_tensor(0).astype(dtype))
                classification_losses.append(cls_loss.sum())
                continue
                
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = paddle.max(IoU, axis=1), paddle.argmax(IoU, axis=1)

            # compute the loss for classification
            targets = paddle.ones_like(classification) * -1
            # 没有梯度，这种方式可能显存爆炸
            targets[IoU_max < 0.4, :] = 0

            positive_indices = IoU_max > 0.5

            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax]

            targets[positive_indices, :] = 0
            targets_numpy = targets.numpy()
            targets_numpy[positive_indices.numpy(), assigned_annotations.numpy()[positive_indices.numpy(), 4].astype(np.int64)] = 1
            # targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            targets = paddle.to_tensor(targets_numpy)

            alpha_factor = paddle.ones_like(targets) * alpha
            alpha_factor = paddle.where(targets == 1., alpha_factor, 1. - alpha_factor)
            focal_weight = paddle.where(targets == 1., 1. - classification, classification)
            focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

            bce = -(targets * paddle.log(classification) + (1.0 - targets) * paddle.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = paddle.zeros_like(cls_loss)
            cls_loss = paddle.where(targets != -1.0, cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / paddle.clip(num_positive_anchors.astype(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = paddle.to_tensor(assigned_annotations.numpy()[positive_indices.numpy(), :])
                anchor_widths_pi = paddle.to_tensor(anchor_widths.numpy()[positive_indices.numpy()])
                anchor_heights_pi = paddle.to_tensor(anchor_heights.numpy()[positive_indices.numpy()])
                anchor_ctr_x_pi = paddle.to_tensor(anchor_ctr_x.numpy()[positive_indices.numpy()])
                anchor_ctr_y_pi = paddle.to_tensor(anchor_ctr_y.numpy()[positive_indices.numpy()])

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = paddle.clip(gt_widths, min=1)
                gt_heights = paddle.clip(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = paddle.log(gt_widths / anchor_widths_pi)
                targets_dh = paddle.log(gt_heights / anchor_heights_pi)

                targets = paddle.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()
                regression_diff = paddle.abs(targets - regression.masked_select(positive_indices.unsqueeze(1).tile([1,regression.shape[1]])).reshape([-1, regression.shape[1]]))
                
                regression_loss = paddle.where(
                    regression_diff <= 1.0 / 9.0,
                    0.5 * 9.0 * paddle.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(paddle.to_tensor(0).astype(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              paddle.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)
            imgs = imgs.transpose([0, 2, 3, 1]).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return paddle.stack(classification_losses).mean(axis=0, keepdim=True), \
               paddle.stack(regression_losses).mean(axis=0, keepdim=True) * 50  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233
