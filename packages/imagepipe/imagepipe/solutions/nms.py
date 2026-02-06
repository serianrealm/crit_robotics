"""Array-based helpers for filtering detections."""
import torch
import torchvision
import numpy as np

def non_max_suppresson(
    prediction: torch.Tensor|np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: torch.Tensor|np.ndarray|None = None,
    max_det: int = 300,
):
    """
    Perform non-maximum suppression (NMS) on prediction results.

    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple detection
    formats including standard boxes, rotated boxes, and masks.

    Args:
        prediction (np.ndarray): Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing boxes, classes, and optional masks.
        conf_thres (float): Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.
        classes (list[int], optional): List of class indices to consider. If None, all classes are considered.
        max_det (int): Maximum number of detections to keep per image.

    Returns: 
        output (list[np.ndarray]): List of detections per image with shape (num_boxes, 6 + num_masks)
            containing (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    prediction = torch.tensor(prediction)

    if classes is not None:
        classes = torch.tensor(classes, dtype=int)
    output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
    if classes is not None:
        output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
    return [out[torchvision.ops.nms(out[:, 0:4], out[:, 4], iou_threshold=iou_thres)] for out in output]
