import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, anchors, strides, loss_weights=None, label_smoothing=0.1,
                 size_penalty_weight=0.15, max_box_ratio=0.4):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.na = len(anchors[0])

        self.size_penalty_weight = size_penalty_weight
        self.max_box_ratio = max_box_ratio

        self.register_buffer('strides', torch.tensor(strides, dtype=torch.float32))

        anchors_tensor = torch.zeros(len(anchors), self.na, 2)
        for i, scale_anchors in enumerate(anchors):
            anchors_tensor[i] = torch.tensor(scale_anchors, dtype=torch.float32)
        self.register_buffer('anchors_tensor', anchors_tensor)

        self.loss_weights = loss_weights or {"box": 0.05, "obj": 0.3, "cls": 0.5}

        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')

        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0


    def focal_loss(self, pred, target, alpha=0.25, gamma=1.5):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def size_penalty_loss(self, predicted_boxes):
        if predicted_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=predicted_boxes.device)

        widths = torch.clamp(predicted_boxes[:, 2], min=1e-6)
        heights = torch.clamp(predicted_boxes[:, 3], min=1e-6)
        areas = widths * heights

        penalties = torch.zeros_like(areas)

        max_area = self.max_box_ratio ** 2
        oversized_mask = areas > max_area
        if oversized_mask.any():
            size_penalty = torch.pow((areas[oversized_mask] - max_area) / (1 - max_area), 2)
            penalties[oversized_mask] += size_penalty

        aspect_ratios = widths / (heights + 1e-6)
        extreme_aspect_mask = (aspect_ratios > 3.0) | (aspect_ratios < 1/3.0)
        if extreme_aspect_mask.any():
            aspect_penalty = torch.abs(torch.log(aspect_ratios[extreme_aspect_mask]))
            penalties[extreme_aspect_mask] += aspect_penalty * 0.5

        full_coverage_mask = (widths > 0.95) | (heights > 0.95)
        if full_coverage_mask.any():
            coverage_penalty = torch.exp(torch.max(widths[full_coverage_mask], heights[full_coverage_mask]) - 0.95) - 1
            penalties[full_coverage_mask] += coverage_penalty * 2.0

        return penalties.mean() if penalties.any() else torch.tensor(0.0, device=predicted_boxes.device)

    def forward(self, predictions, targets):
        device = predictions[0].device

        self.anchors_tensor = self.anchors_tensor.to(device)
        self.strides = self.strides.to(device)

        tcls, tbox, indices, anch = self.build_targets(predictions, targets)

        lcls = torch.tensor(0.0, device=device)
        lbox = torch.tensor(0.0, device=device)
        lobj = torch.tensor(0.0, device=device)
        lsize = torch.tensor(0.0, device=device)

        for i, pred_i in enumerate(predictions):
            b, _, h, w = pred_i.shape

            pred_i = pred_i.view(b, self.na, self.num_classes + 5, h, w).permute(0, 1, 3, 4, 2).contiguous()

            if i < len(indices) and indices[i] and len(indices[i]) == 4:
                b_idx, a_idx, gj, gi = indices[i]
            else:
                empty_tensor_long = torch.empty(0, dtype=torch.long, device=device)
                b_idx, a_idx, gj, gi = empty_tensor_long, empty_tensor_long, empty_tensor_long, empty_tensor_long

            tobj = torch.zeros_like(pred_i[..., 4])

            n = b_idx.shape[0]
            if n > 0:
                if (b_idx.max() >= b or a_idx.max() >= self.na or
                    gj.max() >= h or gi.max() >= w or
                    b_idx.min() < 0 or a_idx.min() < 0 or
                    gj.min() < 0 or gi.min() < 0):
                    continue

                ps = pred_i[b_idx, a_idx, gj, gi]

                pxy = ps[:, :2].sigmoid()
                pwh = torch.clamp(ps[:, 2:4], min=-10, max=10)

                if i < len(anch) and anch[i].shape[0] > 0:
                    predicted_wh = torch.exp(pwh) * anch[i] / self.strides[i] / max(h, w)
                    predicted_wh = torch.clamp(predicted_wh, min=1e-6, max=1.0)
                    predicted_boxes_for_penalty = torch.cat([pxy, predicted_wh], dim=1)
                    lsize += self.size_penalty_loss(predicted_boxes_for_penalty)

                if i < len(tbox) and tbox[i].shape[0] > 0:
                    txy = tbox[i][:, :2]
                    twh = tbox[i][:, 2:4]

                    lbox += F.smooth_l1_loss(pxy, txy, reduction='mean', beta=0.1)
                    lbox += F.smooth_l1_loss(pwh, twh, reduction='mean', beta=0.1)

                tobj[b_idx, a_idx, gj, gi] = 1.0

                if self.num_classes > 1 and i < len(tcls) and tcls[i].shape[0] > 0:
                    target_cls = torch.zeros_like(ps[:, 5:], device=device)
                    if self.label_smoothing > 0:
                        temp_one_hot = torch.zeros_like(target_cls)
                        valid_cls_mask = (tcls[i] >= 0) & (tcls[i] < self.num_classes)
                        if valid_cls_mask.any():
                            valid_indices = torch.arange(len(tcls[i]), device=device)[valid_cls_mask]
                            temp_one_hot[valid_indices, tcls[i][valid_cls_mask]] = 1.0
                        target_cls = self.smooth_labels(temp_one_hot, self.num_classes, self.label_smoothing)
                    else:
                        valid_cls_mask = (tcls[i] >= 0) & (tcls[i] < self.num_classes)
                        if valid_cls_mask.any():
                            valid_indices = torch.arange(len(tcls[i]), device=device)[valid_cls_mask]
                            target_cls[valid_indices, tcls[i][valid_cls_mask]] = 1.0

                    lcls += self.focal_loss(ps[:, 5:], target_cls, self.focal_loss_alpha, self.focal_loss_gamma)

            lobj += self.focal_loss(pred_i[..., 4], tobj, self.focal_loss_alpha, self.focal_loss_gamma)

        total_loss = (lbox * self.loss_weights["box"] +
                      lobj * self.loss_weights["obj"] +
                      lcls * self.loss_weights["cls"] +
                      lsize * self.size_penalty_weight)

        total_loss = torch.clamp(total_loss, max=100.0)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(1.0, device=device, requires_grad=True)
            print("Warning: Loss became NaN/Inf, using fallback value 1.0 for stability.")

        loss_items = torch.stack([lbox.detach(), lobj.detach(), lcls.detach(),
                                 lsize.detach(), total_loss.detach()])

        return total_loss, loss_items

    def smooth_labels(self, targets, num_classes, smoothing=0.1):
        confidence = 1.0 - smoothing
        smooth_value = smoothing / (num_classes - 1)

        return (confidence * targets) + (smooth_value * (1 - targets))


    def build_targets(self, predictions, targets):
        device = predictions[0].device

        self.anchors_tensor = self.anchors_tensor.to(device)
        self.strides = self.strides.to(device)

        if targets.shape[0] == 0:
            return self._create_empty_targets(predictions, device)

        targets = targets.to(device)

        tcls, tbox, indices, anchors_assigned = [], [], [], []

        for i, pred_i in enumerate(predictions):
            b, _, h, w = pred_i.shape
            gain = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)

            scaled_targets = targets.clone()
            scaled_targets[:, 2:6] *= gain

            gxy = scaled_targets[:, 2:4]
            gxi, gyi = gxy.long().T

            valid_mask = (gxi >= 0) & (gxi < w) & (gyi >= 0) & (gyi < h)

            if not valid_mask.any():
                self._append_empty_targets(indices, tbox, tcls, anchors_assigned, device)
                continue

            t_valid = scaled_targets[valid_mask]
            gxi_valid = gxi[valid_mask]
            gyi_valid = gyi[valid_mask]

            b_idx = t_valid[:, 0].long()
            c_idx = t_valid[:, 1].long()

            gwh = t_valid[:, 4:6]
            num_targets = t_valid.shape[0]

            anchors_for_scale = self.anchors_tensor[i].to(device)

            ratios = gwh.unsqueeze(1) / (anchors_for_scale.unsqueeze(0) + 1e-16)
            best_ratios = torch.max(ratios, 1.0 / (ratios + 1e-16)).max(2)[0]

            best_anchor_idx = best_ratios.argmin(1)

            indices.append((
                b_idx.detach().clone(),
                best_anchor_idx.detach().clone(),
                gyi_valid.detach().clone(),
                gxi_valid.detach().clone()
            ))

            target_boxes_i = torch.zeros_like(t_valid[:, 2:6])
            target_boxes_i[:, :2] = gxy[valid_mask] - torch.stack([gxi_valid, gyi_valid], 1).float()
            target_boxes_i[:, 2:4] = torch.log(gwh + 1e-16) - torch.log(anchors_for_scale[best_anchor_idx] + 1e-16)

            tbox.append(target_boxes_i)
            tcls.append(c_idx)
            anchors_assigned.append(anchors_for_scale[best_anchor_idx])

        return tcls, tbox, indices, anchors_assigned


    def _append_empty_targets(self, indices, tbox, tcls, anchors, device):
        empty_tensor_long = torch.empty(0, dtype=torch.long, device=device)
        empty_tensor_float4 = torch.empty(0, 4, device=device)
        empty_tensor_float2 = torch.empty(0, 2, device=device)

        indices.append((empty_tensor_long, empty_tensor_long, empty_tensor_long, empty_tensor_long))
        tbox.append(empty_tensor_float4)
        tcls.append(empty_tensor_long)
        anchors.append(empty_tensor_float2)

    def _create_empty_targets(self, predictions, device):
        tcls, tbox, indices, anchors = [], [], [], []
        for _ in predictions:
            self._append_empty_targets(indices, tbox, tcls, anchors, device)
        return tcls, tbox, indices, anchors