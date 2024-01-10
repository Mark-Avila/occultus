import unittest
import sys
import os
import shutil

sys.path.append("C:\programming\occultus")
from occultus.core import Occultus


class TestDetectVideo(unittest.TestCase):
    def test_video(self):
        video_name = "mememe.mp4"
        occultus = Occultus(
            "weights/kamukha-v3.pt",
            output_folder="test_output",
            output_create_folder=False,
        )
        try:
            occultus.detect_video(f"video/{video_name}")
        except:
            self.fail("Failed to detect video")
        self.assertTrue(os.path.exists(f"test_output/{video_name}"))
        shutil.rmtree("test_output")

    def test_invalid_path(self):
        occultus = Occultus("weights/kamukha-v3.pt")
        with self.assertRaises(Exception):
            occultus.detect_video("jsahdkjahsdjsahdkja")

    def test_invalid_video(self):
        occultus = Occultus("weights/kamukha-v3.pt")
        with self.assertRaises(Exception):
            occultus.detect_video("video/group.jpg")


if __name__ == "__main__":
    unittest.main()
