import unittest
import sys

sys.path.append("C:\programming\occultus")

from occultus.core import Occultus


class TestConstructor(unittest.TestCase):
    def test_initialization(self):
        occultus = Occultus("weights/kamukha-v3.pt")
        self.assertIsInstance(occultus, Occultus)

    def test_config(self):
        occultus = Occultus(
            "weights/kamukha-v3.pt",
            conf_thres=0.5,
            iou=0.45,
            device="0",
            img_size=640,
            show_label=True,
            show_track=True,
            output_folder="test_output",
            output_create_folder=True,
            blur_type="pixel",
            select_type="all",
            id_list=[1],
            reset_kalman=True,
            intensity=51,
        )
        self.assertIsInstance(occultus, Occultus)

    def test_invalid_weights(self):
        with self.assertRaises(FileNotFoundError):
            _ = Occultus("ashdkjasashdk")

    def test_invalid_config(
        self,
    ):
        with self.assertRaises(Exception):
            _ = Occultus(
                "weights/kamukha-v3.pt",
                conf_thres="adsad",
                iou="asdasd",
                img_size="asdasda",
                show_label="asdasda",
                show_track="asdasda",
                blur_type="asdasd",
                select_type="asdasda",
                id_list=["asdasda"],
                reset_kalman="asdasda",
                intensity="asdasda",
                output_folder="test_output",
                output_create_folder="asdasda",
                device=0,
            )


if __name__ == "__main__":
    unittest.main()
