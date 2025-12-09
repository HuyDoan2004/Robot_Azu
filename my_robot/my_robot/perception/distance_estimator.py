#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class DistanceEstimator:
    def __init__(self, kernel=11):
        # kernel nên là số lẻ: 5/7/11. 11 cho kết quả ổn định hơn khi nhiễu depth.
        self.kernel = int(kernel) if int(kernel) > 0 else 11

    def estimate(self, depth_img, dets):
        """Trả về list độ sâu theo mm ứng với mỗi detection.
        - depth_img: numpy array (16UC1 mm hoặc 32FC1 m)
        - dets: list[dict] có key 'bbox'=[x1,y1,x2,y2]
        """
        if depth_img is None or dets is None:
            return []

        H, W = depth_img.shape[:2]
        mm_list = []

        # xác định kiểu dữ liệu depth
        is_u16 = depth_img.dtype == np.uint16

        for d in dets:
            x1, y1, x2, y2 = map(int, d.get('bbox', [0, 0, 0, 0]))
            cx = max(0, min(W - 1, (x1 + x2) // 2))
            cy = max(0, min(H - 1, (y1 + y2) // 2))

            k = self.kernel
            x0 = max(0, cx - k // 2); x1c = min(W, cx + k // 2 + 1)
            y0 = max(0, cy - k // 2); y1c = min(H, cy + k // 2 + 1)
            roi = depth_img[y0:y1c, x0:x1c]

            if is_u16:
                valid = roi[roi > 0]
                mm = int(np.median(valid)) if valid.size else 0
            else:  # float (m)
                valid = roi[np.isfinite(roi) & (roi > 0.0)]
                mm = int(float(np.median(valid)) * 1000.0) if valid.size else 0

            mm_list.append(mm)
        return mm_list
