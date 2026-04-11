#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

class Visualizer:
    def __init__(self, show=False, show_id=False):
        self.show = bool(show)
        self.show_id = bool(show_id)

    def draw(self, bgr, dets, depths_mm=None):
        img = bgr
        if dets is None:
            return img

        for i, d in enumerate(dets):
            x1, y1, x2, y2 = map(int, d.get('bbox', [0, 0, 0, 0]))
            conf = float(d.get('conf', 0.0))
            cls  = d.get('cls', -1)
            name = d.get('name', cls)  # <<< ƯU TIÊN tên lớp
            tid  = int(d.get('id', -1))

            # khung
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 255), 2)

            # khoảng cách (m) nếu có
            dist_txt = ''
            if depths_mm is not None and i < len(depths_mm) and depths_mm[i]:
                try:
                    dist_m = float(depths_mm[i]) / 1000.0
                    if dist_m > 0:
                        dist_txt = f' {dist_m:.2f} m'
                except Exception:
                    pass

            # label text
            base = f'{name} {conf:.2f}{dist_txt}'
            text = f'ID {tid} | {base}' if self.show_id and tid != -1 else base

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y0 = max(0, y1 - th - 6)
            cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y0 + th + 6), (0, 230, 255), -1)
            cv2.putText(img, text, (x1 + 3, y0 + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        return img
