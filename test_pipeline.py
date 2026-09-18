import unittest
import numpy as np

class TestAugmentationPipeline(unittest.TestCase):
    def setUp(self):
        self.img = np.ones((100, 100, 3), dtype=np.uint8) * 100

    def test_brightness_shape(self):
        res = np.clip(self.img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
        self.assertEqual(res.shape, (100, 100, 3))
        self.assertEqual(res.dtype, np.uint8)

    def test_negative_values(self):
        res = (255 - self.img).astype(np.uint8)
        self.assertEqual(int(res[0, 0, 0]), 155)

    def test_rotation_shape(self):
        res = np.rot90(self.img)
        self.assertEqual(res.shape, (100, 100, 3))

if __name__ == "__main__":
    unittest.main()
