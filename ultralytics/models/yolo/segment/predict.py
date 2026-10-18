# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import numpy as np
import torch
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    The paper's binary image restoration framework (skeletonization, direction-aware endpoint
    connection, line-width restoration) is available but disabled by default; enable it by
    setting `predictor.refine_line_masks = True`.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolo11n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"
        self.line_width = 4
        # Paper's binary image restoration framework (line-mask refinement); disabled by default
        self.refine_line_masks = False

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression, refines line masks and processes detections for each image in a batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            if masks is not None and self.refine_line_masks:
                masks = self.refine_masks(masks)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def refine_masks(self, masks):
        """Refine predicted line masks: skeletonize, connect endpoints, then restore line width."""
        refined = [
            self.restore_line_width(self.connect_endpoints(self.skeletonize_mask(m)))
            for m in masks.detach().cpu().numpy()
        ]
        return torch.from_numpy(np.stack(refined)).to(masks.device)

    def skeletonize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Binarize, close small holes, then thin the mask to a one-pixel skeleton."""
        _, binary = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return skeletonize(closed).astype(np.uint8)

    def find_endpoints(self, skeleton: np.ndarray) -> list:
        """Detect skeleton endpoints (points with exactly one neighbor in the 8-neighborhood)."""
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        conv = cv2.filter2D(skeleton, -1, kernel)
        return [(y, x) for y, x in zip(*np.where(conv == 11))]  # center 10 + single neighbor 1 = 11

    def connect_endpoints(self, skeleton: np.ndarray, max_distance=30, angle_tolerance=70) -> np.ndarray:
        """Connect nearby endpoints whose local line directions are consistent."""
        endpoints = self.find_endpoints(skeleton)
        if len(endpoints) < 2:
            return skeleton

        kdtree = KDTree(endpoints)
        connected = skeleton.copy()
        for i, j in kdtree.query_pairs(max_distance):
            pt1, pt2 = endpoints[i], endpoints[j]
            if self.is_direction_consistent(pt1, pt2, skeleton, angle_tolerance):
                cv2.line(connected, tuple(reversed(pt1)), tuple(reversed(pt2)), 1, thickness=1)
        return connected

    def is_direction_consistent(self, pt1, pt2, skeleton, angle_tolerance, window=5) -> bool:
        """Check that the local line direction at both endpoints agrees with the connection direction."""
        y1, x1 = pt1
        y2, x2 = pt2
        roi1 = skeleton[max(0, y1 - window) : y1 + window + 1, max(0, x1 - window) : x1 + window + 1]
        roi2 = skeleton[max(0, y2 - window) : y2 + window + 1, max(0, x2 - window) : x2 + window + 1]

        def get_direction(roi):
            """Return the dominant contour orientation in degrees, or None if unavailable."""
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            line_params = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
            return np.degrees(np.arctan2(line_params[1], line_params[0])) % 180

        angle1 = get_direction(roi1)
        angle2 = get_direction(roi2)
        if angle1 is None or angle2 is None:
            return False

        connect_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

        def diff(a, b):
            """Angular difference on the 0-180 circle."""
            return min(abs(a - b), 180 - abs(a - b))

        return diff(angle1, connect_angle) < angle_tolerance and diff(angle2, (connect_angle + 180) % 180) < angle_tolerance

    def restore_line_width(self, skeleton: np.ndarray) -> np.ndarray:
        """Dilate the refined skeleton back to the target line width."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        iterations = max(1, self.line_width // 2)
        return cv2.dilate(skeleton, kernel, iterations=iterations)
