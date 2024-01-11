import unittest
import sys
import cv2
import numpy as np

sys.path.append("C:\programming\occultus")

from occultus.core import Occultus


class TestDetectInput(unittest.TestCase):
    def test_input_record(self):
        occultus = Occultus("weights/kamukha-v3.pt")

        try:
            for frame, boxes in occultus.detect_input_generator():
                self.assertIsInstance(frame, np.ndarray)
                self.assertIsInstance(boxes, list)
                break
        except:
            self.fail()

    def test_invalid_device(self):
        occultus = Occultus("weights/kamukha-v3.pt")
        with self.assertRaises(Exception):
            occultus.detect_input("jsahdkjahsdjsahdkja")


if __name__ == "__main__":
    unittest.main()
