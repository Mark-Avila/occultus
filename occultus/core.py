"""
Name: Occultus
Date: Jan 5, 2024
File Description: The fundamental features and components of Occultus.
Version: 0.2

Developer/Author: Mark Christian A. Avila
QA tester: John Remmon G. Castor
Section/Course: BSCS-NS-4B
Description: Face detection library with privacy controls: blurring, exclusion, and specific face detection. Capstone Project for Bachelor of Science in Computer Science (Non-stem)

Modification History:
- December Jan 5, 2024 (ver 0.2):
    - ver 1.0 of the full interface
        - Upload, Detect, and render videos
        - Record live camera with detection
        - Able to upload online video URL's
- December 12, 2023 (ver 0.1):
    - First working version that has the project requirements
        - All, Specific, and Exclusion censoring
        - Click/Select boxes to apply privacy controls
        - Able to Upload videos and use live feed 
    - Censor type handling
    - Privacy control handling
    - Documentation
    - Seperated model initialization, inference, and post process into methods

Planned:
- Live recording links/URL's
- Select face in recording via click
- Live preview of selected options in video
- Interface Improvements
"""

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import datetime

from occultus.models.experimental import attempt_load
from occultus.utils.datasets import letterbox
from occultus.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
    increment_path,
)
from occultus.utils.plots import draw_boxes, blur_boxes, pixelate_boxes, fill_boxes
from occultus.utils.torch_utils import select_device
from occultus.utils.sort import *
import sys

sys.path.insert(0, "./occultus")


class Occultus:
    def __init__(
        self,
        weights,
        conf_thres=0.25,
        iou=0.45,
        device="",
        img_size=640,
        show_track=False,
        show_label=False,
        output_folder="output",
        output_name=None,
        output_create_folder=True,
        blur_type="detect",
        select_type="all",
        id_list=[],
        reset_kalman=True,
        intensity=51,
    ):
        # Essential attributes
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou = iou  # ///
        self.device = device
        self.output = output_folder  # ///
        self.name = (
            datetime.datetime.now().strftime("%b_%d_%Y-%H_%M_%S_")
            if output_name is None
            else output_name
        )
        self.show_track = show_track
        self.blur_type = blur_type
        self.select_type = select_type
        self.id_list: list = id_list
        self.intensity = intensity

        self.flipped = False
        self.nobbox = False
        self.nolabel = not show_label
        self.model = {}

        set_logging()

        self.save_dir = (
            Path(increment_path(Path(self.output) / self.name, exist_ok=False))
            if output_create_folder
            else Path(self.output)
        )

        self.tracker = Sort()

        if reset_kalman:
            # Resets tracking ID's to 0
            KalmanBoxTracker.count = 0

        self.augment = False
        self.device = select_device(device)
        self.half = self.device.type != "cpu"
        self.model = attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)

        self.model.half()

        # Run inference once
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.img_size, self.img_size)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once

    def add_id(self, new_id: int) -> int:
        """Append ID value for privacy controls (specific and exclusion)

        Args:
            new_id (int): New ID
        """
        if isinstance(new_id, int):
            self.id_list.append(new_id)
        else:
            raise ValueError("Invalid ID to append, must be of type integer")

    def remove_id(self, id: int):
        """Pops specified from the ID list for privacy controls (specific and exclusion)

        Args:
            id (int): ID to remove/pop
        """
        if isinstance(id, int):
            if id in self.id_list:
                self.id_list.remove(id)
        else:
            raise ValueError("Invalid ID to pop, must be of type integer")

    def set_blur_type(
        self, new_type: str = "default", show_label=False, show_track=False
    ):
        """Set type of blurring to be applied (default, gaussian, pixel, fill)

        Args:
            new_type (str, optional): Blurring type. Defaults to "default".
        """
        if (
            new_type == "default"
            or new_type == "gaussian"
            or new_type == "pixel"
            or new_type == "fill"
        ):
            self.show_track = show_track
            self.nolabel = not show_label
            self.blur_type = new_type
        else:
            raise ValueError(
                f"Invalid censor type '{new_type}'. Please choose between the following: [default, gaussian, pixel, fill]"
            )

    def set_privacy_control(self, new_type: str = "all"):
        """
        Set the privacy control mode for object detection results.

        Parameters:
        - new_type (str): The privacy control mode. It can be one of the following:
        - 'all': Apply privacy control to all detected objects.
        - 'specific': Apply privacy control only to objects specified in the `id_list`.
        - 'exclude': Exclude objects specified in the `id_list` from privacy control.

        Example:
        ```python
        instance_of_your_class.set_privacy_control(new_type="specific")
        ```

        Raises:
        - ValueError: If an invalid `new_type` is provided. Valid options are ['all', 'specific', 'exclude'].

        Note:
        - This method allows you to set the privacy control mode for object detection results.
        - The privacy control mode determines whether to apply blurring or other effects to all detected
        objects, specific objects, or exclude specific objects.
        - The `new_type` parameter should be one of the valid options: 'all', 'specific', or 'exclude'.
        - If an invalid option is provided, a `ValueError` is raised.
        """
        if new_type == "specific" or new_type == "exclude" or new_type == "all":
            self.select_type = new_type
        else:
            raise ValueError(
                f"Invalid select mode '{new_type}'. Please choose between the following: ['all', 'specific', 'exclude']"
            )

        return

    def detect(self, frame):
        """
        This method takes a cv2 frame as input, applies face detection using the specified pre-trained model,
        and will returns the processed frame with applied detections and blurring. The method also provides
        information about the detected faces, including their tracking IDs and bounding box coordinates.

        Parameters:
        - frame: cv2 Frame

        Example:
        ```python
        occultus = Occultus("path/to/weights.pt", ...args)
        cap = VideoCapture(0)
        ret_val, frame = cap.read()
        detected_frame, boxes = occultus.detect(frame)
        cv2.imshow("Detected Frame", detected_frame)
        ```

        Returns:
        - detected_frame: cv2 Frame with applied detections and blurring
        - boxes: List of dictionaries, each dict containing the following values:
            - ID: Object tracking ID
            - Box: Object Boxes (coordinates of the detected face)
        """

        img = self.__to_ndarray(frame)
        img = self.__preprocess(img)
        pred = self.__inference(img)
        results = self.__postprocess(pred, img, frame)

        # [frame, bboxes] = results
        return results

    def detect_image(self, source, intensity=51):
        """
        A method for quick face detection and blurring from an image source

        Parameters:
        - source (str): Image file path

        Example:
        ```python
        occultus = Occultus("path/to/weights.pt", ...args)
        occultus.detect_image("path/to/image.jpg")
        ```

        Returns:
            - image: cv2 frame of image with face detection and blurring
        """
        og_img = cv2.imread(source)

        if not og_img.isOpened():
            raise ValueError("Failed to load image")

        img = self.__to_ndarray(og_img)
        img = self.__preprocess(img)
        pred = self.__inference(img)
        [result_img, bboxes] = self.__postprocess(pred, img, og_img)

        new_width = 800
        new_height = 600
        result_img = cv2.resize(result_img, (new_width, new_height))

        cv2.imshow("Occultus", result_img)
        cv2.waitKey(0)  # 1 millisecond

        return result_img

    def detect_video(self, source):
        """
        A method for quick face detection and blurring from a video source

        Parameters:
        - source: cv2 Frame

        Output:
        - MP4 video with face detection and blurring to the output folder
        """
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise FileNotFoundError("Could not find video")

        vid_formats = [
            ".mov",
            ".avi",
            ".mp4",
            ".mpg",
            ".mpeg",
            ".m4v",
            ".wmv",
            ".mkv",
        ]

        vid_format = os.path.splitext(os.path.basename(source))[1]

        if vid_format not in vid_formats:
            raise ValueError("Invalid video format")

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_dir = self.save_dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(source))

        writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(width), int(height)),
        )

        frame_id = 1
        while True:
            ret_val, og_img = cap.read()

            if not ret_val:
                break

            img = self.__to_ndarray(og_img)
            img = self.__preprocess(img)
            pred = self.__inference(img)
            [result_img, _] = self.__postprocess(pred, img, og_img)
            writer.write(result_img)

            print(f"Frame {frame_id}/{frame_count}: Done")

            frame_id = frame_id + 1

        writer.release()
        cv2.destroyAllWindows()

    def detect_video_generator(self, source, save_video: bool = True):
        """
        Generator method similar to the ``detect_video()`` function, providing additional video data for further processing.

        Parameters:
            - ``source`` (str): The file path of the input video.
            - ``save_video`` (bool, optional): Flag indicating whether to save the processed video. Default is False.

        Example:
            ```python
            occultus = Occultus("path/to/weights.pt", ...args)

            for bboxes, frame_id, frame_count in occultus.detect_video_generator("video/path.mp4"):
                print("Bounding Boxes:", bboxes, " | Object ID:", frame_id, " | Frame Count:", frame_count)
            ```

        Yields:
            - ``bboxes`` (list): List of bounding box coordinates for the current frame.
            - ``frame_id``: Object ID for the current frame.
            - ``frame_count``: Current frame iteration count.
        """

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError("Failed to load source")

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if save_video:
            output_dir = self.save_dir
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, os.path.basename(source))

            writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (int(width), int(height)),
            )

        frame_id = 1
        while True:
            ret_val, og_img = cap.read()

            if not ret_val:
                break

            img = self.__to_ndarray(og_img)
            img = self.__preprocess(img)
            pred = self.__inference(img)
            [result_img, bboxes] = self.__postprocess(pred, img, og_img)

            if save_video:
                writer.write(result_img)

            yield bboxes, frame_id, frame_count

            frame_id = frame_id + 1

        if save_video:
            writer.release()
        cv2.destroyAllWindows()

    def detect_input(self, source: str = "0", frame_interval=2):
        """
        A quick method for detecting and blurring faces from an input device.

        Parameters:
        - ``source`` (str): The file path of the input video. Defaults to ``0`` for the systems' internal camera (Will not work if it doesn't have one)
        - ``frame_interval`` (number, optional): Number of frames to skip before grabbing another frame for detection. Defaults to ``2``

        Example:
        ```
        from occultus.core import Occultus
        occultus = Occultus("path/to/weights.pt", ...args)

        occultus.detect_input("0")
        ```
        """
        if source.isnumeric():
            source = eval(source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError("Failed to load source")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        while True:
            if cv2.waitKey(1) == ord("q"):
                break

            if source == 0:
                ret_val, og_img = cap.read()
                og_img = cv2.flip(og_img, 1)
            else:
                n = 0
                while True:
                    n += 1
                    cap.grab()
                    if n == frame_interval:  # Grab every frame by frame_interval
                        ret_val, og_img = cap.retrieve()
                        n = 0
                        if ret_val:
                            break

            assert ret_val, f"Camera Error {source}"

            img = self.__to_ndarray(og_img)
            img = self.__preprocess(img)
            pred = self.__inference(img)
            [result_img, bboxes] = self.__postprocess(pred, img, og_img)

            print(bboxes)

            cv2.imshow("Occultus", result_img)

        cap.release()
        cv2.destroyAllWindows()

    def detect_input_generator(self, source: str = "0", frame_interval=2):
        """
        A method similar to ``detect_input()``, providing additional video data for further processing

        Parameters:
        - ``source`` (str): The file path of the input video. Defaults to ``0`` for the systems' internal camera (Will not work if it doesn't have one)
        - ``frame_interval`` (number, optional): Number of frames to skip before grabbing another frame for detection. Defaults to ``2``

        Example:
        ```python
        from occultus.core import Occultus
        occultus = Occultus("path/to/weights.pt", ...args)
        for frame_id, boxes in occultus.detect_input_generator():
            print(boxes)
            print(frame_id)
        ```
        """
        if source.isnumeric():
            source = eval(source)

        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        try:
            while True:
                if source == 0:
                    ret_val, og_img = cap.read()
                    og_img = cv2.flip(og_img, 1)
                else:
                    n = 0
                    while True:
                        n += 1
                        cap.grab()
                        if n == frame_interval:  # Grab every frame by frame_interval
                            ret_val, og_img = cap.retrieve()
                            n = 0
                            if ret_val:
                                break

                assert ret_val, f"Camera Error {source}"

                img = self.__to_ndarray(og_img)
                img = self.__preprocess(img)
                pred = self.__inference(img)
                [frame, bboxes] = self.__postprocess(pred, img, og_img)

                # [frame, bboxes] = results
                yield frame, bboxes
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def __to_ndarray(self, frame):
        # Padded resize
        img = letterbox(frame, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img

    def __preprocess(self, img):
        # Padded resize
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def __inference(self, img):
        old_img_w = old_img_h = self.img_size
        old_img_b = 1

        # Warmup
        if self.device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        # Inference
        # t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]
        # t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou,
            classes=0,
            agnostic=False,
        )

        return pred

    def __postprocess(self, preds, img, og_img):
        bboxes = []

        for i, det in enumerate(preds):
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], og_img.shape
                ).round()

                s = ""

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} Face{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0, 6))

                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack(
                        (dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass]))
                    )

                tracked_dets = self.tracker.update(dets_to_sort, False)
                tracks = self.tracker.getTrackers()

                # draw boxes for visualization
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    confidences = None

                    if self.show_track:
                        # loop over tracks
                        for t, track in enumerate(tracks):
                            track_color = (0, 0, 255)

                            [
                                cv2.line(
                                    og_img,
                                    (
                                        int(track.centroidarr[i][0]),
                                        int(track.centroidarr[i][1]),
                                    ),
                                    (
                                        int(track.centroidarr[i + 1][0]),
                                        int(track.centroidarr[i + 1][1]),
                                    ),
                                    track_color,
                                    thickness=2,
                                )
                                for i, _ in enumerate(track.centroidarr)
                                if i < len(track.centroidarr) - 1
                            ]
                else:
                    bbox_xyxy = dets_to_sort[:, :4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]

                names = (
                    self.model.names
                    if hasattr(self.model, "module")
                    else self.model.names
                )

                blur_functions = {
                    "gaussian": blur_boxes,
                    "pixel": pixelate_boxes,
                    "fill": fill_boxes,
                    "detect": draw_boxes,
                }

                blur_function = blur_functions.get(self.blur_type, draw_boxes)

                result_img = blur_function(
                    og_img,
                    bbox_xyxy,
                    ids=identities,
                    ids_list=self.id_list,
                    privacy=self.select_type,
                    categories=categories,
                    confidences=confidences,
                    names=names,
                    nobbox=self.nobbox,
                    nolabel=self.nolabel,
                    intensity=self.intensity,
                )

                new_preds = None

                if identities is not None:
                    for index, idt in enumerate(identities):
                        new_preds = {"id": int(idt), "box": bbox_xyxy[index]}
                        bboxes.append(new_preds)

            else:
                result_img = og_img

        return [result_img, bboxes]

    def get_ids(self):
        return self.id_list


"""
TODO: Test cases

Occultus constructor - Occultus():
- POSITIVE TEST CASES:
    1. Basic Initialization:
        - Provide valid values for weights.
        - Use default values for all other parameters.
    2. Custom Configuration:
        - Provide valid values for weights.
        - Set parameters 
            - conf_thres 
            - iou
            - device
            - img_size
            - show_track
            - show_label
            - output_folder
            - output_name
            - output_create_folder
            - blur_type
            - select_type
            - id_list
            - reset_kalman 
            - intensity
            - Set output_folder and output_name to specific values.

- NEGATIVE TEST CASES:
    1. Invalid Weights:
    2. Invalid Configuration:

--------------

detect_image():

POSITIVE TEST CASES:
    1. Put image file path

NEGATIVE TEST CASES:
    1. Invalid image file path
    2. Invalid image type (webp)

--------------

detect_video():

POSITIVE TEST CASES:
    1. Put video file path
    2. Check video in output folder with date of today

NEGATIVE TEST CASES:
    1. Invalid video file path
    2. Invalid type (.jpg, .png)

--------------

detect_input():

POSITIVE TEST CASES:
    1. Put "0" (string) in the source
    2. Check video in output folder with date of today

NEGATIVE TEST CASES:
    1. Invalid video file path
    2. Invalid type (.jpg, .png)
    3. number source (0 instead of "0")
    
--------------

append_id() / pop_id():
TEST CASE PROCEDURE:
    1. initiate occultus
    2. add number (3) with append_id
    2. remove number (3) with pop_id

---------------

Stream Interface:

POSITIVE TEST CASES:
    1. with blur type "default" and privacy "all"
    2. with blur type "default" and privacy "specific"
    3. with blur type "default" and privacy "exclude"
    4. with blur type "pixel" and privacy "all"
    5. with blur type "pixel" and privacy "specific"
    6. with blur type "pixel" and privacy "exclude"
    7. with blur type "gaussian" and privacy "all"
    8. with blur type "gaussian" and privacy "specific"
    9. with blur type "gaussian" and privacy "exclude"
    10. with blur type "fill" and privacy "all"
    11. with blur type "fill" and privacy "specific"
    12. with blur type "fill" and privacy "exclude"

NEGATIVE TEST CASES:
    1. Invalid blur type (number, boolean, etc.)
    2. wrong blur type (not "default", "pixel", "gaussian", or "fill")
    3. Empty blur type
    1. Invalid privacy type (number, boolean, etc.)
    2. wrong privacy type (not "all", "specific", "exclude")
    3. Empty privacy type
    

all must expect a return type of "frame" which is a cv2 frame and a list of dictionaries "boxes" 

"""
