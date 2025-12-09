#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

class YoloTracker:
    def __init__(self, model_weights='yolo11n.pt', imgsz=640, use_gpu=True, use_fp16=True):
        """
        Hỗ trợ tự động phát hiện định dạng mô hình:
        - .pt: PyTorch
        - .engine: TensorRT (đã export bằng ultralytics)
        """
        from ultralytics import YOLO
        import torch

        self.model_path = model_weights
        self.imgsz = int(imgsz)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.fp16 = bool(use_fp16 and self.use_gpu)
        self.device = 'cuda' if self.use_gpu else 'cpu'

        # ===== Load Model =====
        if model_weights.endswith('.engine'):
            print(f"[YoloTracker] Loading TensorRT engine: {model_weights}")
            self.model = YOLO(model_weights)  # ultralytics tự nhận TensorRT backend
        elif model_weights.endswith('.pt'):
            print(f"[YoloTracker] Loading PyTorch model: {model_weights}")
            self.model = YOLO(model_weights)
        else:
            raise ValueError(f"Unsupported model format: {model_weights}")

        # ===== GPU / FP16 diagnostics =====
        if self.use_gpu:
            print(f"[YoloTracker] Using CUDA device: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
            if self.fp16:
                print("[YoloTracker] FP16 mode enabled.")
        else:
            print("[YoloTracker] Using CPU inference.")

        # ===== Names of classes =====
        self.names = getattr(self.model, 'names', {})

    def infer(self, bgr_image):
        """Trả về (result, dets_list) – dets_list là list[dict] cho Visualizer/DistanceEstimator."""
        # Chuyển BGR → RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Inference (track persist=True giúp duy trì ID)
        result = self.model.track(
            source=rgb,
            imgsz=self.imgsz,
            device=self.device,
            half=self.fp16,
            persist=True,
            stream=False,
            verbose=False
        )[0]

        boxes = getattr(result, 'boxes', None)
        dets = []
        if boxes is None or len(boxes) == 0:
            return result, dets

        # Tensor → numpy
        xyxy = boxes.xyxy.cpu().numpy().astype(float)
        conf = boxes.conf.cpu().numpy().astype(float)
        cls  = boxes.cls.cpu().numpy().astype(int)
        ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)

        for bb, cf, ci, tid in zip(xyxy, conf, cls, ids):
            x1, y1, x2, y2 = map(int, bb)
            dets.append({
                'bbox': [x1, y1, x2, y2],
                'conf': float(cf),
                'cls': int(ci),
                'id': int(tid),
                'name': self.names[ci] if isinstance(self.names, dict) and ci in self.names else str(ci)
            })

        return result, dets
