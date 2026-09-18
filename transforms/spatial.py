import cv2
import numpy as np

class SpatialTransforms:
    @staticmethod
    def brightness(img: np.ndarray, beta: int = 50) -> np.ndarray:
        return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)

    @staticmethod
    def contrast(img: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    @staticmethod
    def negative(img: np.ndarray, L: int = 256) -> np.ndarray:
        return (L - 1 - img.astype(np.int16)).clip(0, 255).astype(np.uint8)

    @staticmethod
    def log_transform(img: np.ndarray, c: float = None) -> np.ndarray:
        img_f = img.astype(np.float32)
        if c is None:
            c = 255.0 / np.log(1 + np.max(img_f))
        out = c * np.log(1 + img_f)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def gamma(img: np.ndarray, gamma: float = 0.5, c: float = 1.0) -> np.ndarray:
        img_n = img.astype(np.float32) / 255.0
        out = c * np.power(img_n, gamma)
        out = np.clip(out * 255.0, 0, 255)
        return out.astype(np.uint8)

    @staticmethod
    def threshold(img: np.ndarray, thresh: int = 127, maxval: int = 255) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, out = cv2.threshold(gray, thresh, maxval, cv2.THRESH_BINARY)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
