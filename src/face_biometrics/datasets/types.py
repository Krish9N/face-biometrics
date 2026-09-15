from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class FacialLandmarks:
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    left_mouth: tuple[float, float]
    right_mouth: tuple[float, float]


@dataclass
class FaceAnnotation:
    bbox: BoundingBox
    landmarks: Optional[FacialLandmarks] = None
    identity: Optional[int] = None


@dataclass
class FaceSample:
    image_path: str
    faces: list[FaceAnnotation]