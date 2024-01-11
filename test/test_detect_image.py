import unittest
import sys
import cv2

sys.path.append("C:\programming\occultus")

from occultus.core import Occultus


class TestDetectImage(unittest.TestCase):
    def test_image(self):
        occultus = Occultus("weights/kamukha-v3.pt")

        try:
            occultus.detect_image("video/group.jpg")
        except Exception as e:
            self.fail("Failed to detect image")

    def test_frame(self):
        occultus = Occultus("weights/kamukha-v3.pt")

        image = cv2.imread("video/group.jpg")

        try:
            occultus.detect(image)
        except Exception as e:
            self.fail("Failed to detect image")

    def test_invalid_path(self):
        occultus = Occultus("weights/kamukha-v3.pt")
        with self.assertRaises(Exception):
            occultus.detect_image("jsahdkjahsdjsahdkja")

    def test_invalid_image(self):
        occultus = Occultus("weights/kamukha-v3.pt")
        with self.assertRaises(Exception):
            occultus.detect_image("video/giftest.gif")


if __name__ == "__main__":
    unittest.main()
