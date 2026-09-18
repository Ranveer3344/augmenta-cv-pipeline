import cv2
import numpy as np

class GeometricTransforms:
    @staticmethod
    def translate(img: np.ndarray, tx: int = 60, ty: int = 40) -> np.ndarray:
        h, w = img.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def rotate(img: np.ndarray, angle: float = 45.0, scale: float = 1.0) -> np.ndarray:
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def reflect_horizontal(img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 1)

    @staticmethod
    def reflect_vertical(img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 0)

    @staticmethod
    def shear_x(img: np.ndarray, shx: float = 0.3) -> np.ndarray:
        h, w = img.shape[:2]
        M = np.float32([[1, shx, 0], [0, 1, 0]])
        new_w = int(w + abs(shx) * h)
        out = cv2.warpAffine(img, M, (new_w, h))
        return cv2.resize(out, (w, h))

    @staticmethod
    def affine(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        dst_pts = np.float32([[w * 0.1, h * 0.2], [w * 0.9, h * 0.05], [w * 0.15, h * 0.9]])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        return cv2.warpAffine(img, M, (w, h))
