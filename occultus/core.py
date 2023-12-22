"""
Name: Occultus
Date: December 12, 2023
File Description: The fundamental features and components of Occultus.
Version: 0.1

Developer/Author: Mark Christian A. Avila
QA tester: John Remmon G. Castor
Section/Course: BSCS-NS-4B
Description: Face detection library with privacy controls: blurring, exclusion, and specific face detection. Capstone Project for Bachelor of Science in Computer Science (Non-stem)

Modification History:
- December 12, 2023 (ver 0.1):
    - First working version that has the project requirements
        - All, Specific, and Exclusion censoring
        - Click/Select boxes to apply privacy controls
        - Able to Upload videos and use live feed 
    - Censor type handling
    - Privacy control handling
    - Documentation
    - Seperated model initialization, inference, and post process into methods
"""

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import datetime

from PIL import ImageTk
from occultus.models.experimental import attempt_load
from occultus.utils.datasets import letterbox
from occultus.utils.general import (
    check_img_size,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    set_logging,
    increment_path,
)
from occultus.utils.plots import draw_boxes, blur_boxes, pixelate_boxes, fill_boxes
from occultus.utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)

from occultus.utils.sort import *

import sys

sys.path.insert(0, "./occultus")


class Occultus:
    def __init__(
        self,
        weights: str,
        conf_thres=0.25,
        iou=0.45,
        device="",
        img_size=640,
        show_track=False,
    ):
        # Essential attributes
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou = iou  # ///
        self.device = device
        self.output = "output"  # ///
        self.name = datetime.datetime.now().strftime("%b_%d_%Y-%H_%M_%S_")
        self.show_track = show_track
        self.blur_type = "detect"
        self.select_type = "all"

        self.flipped = False
        self.nobbox = False
        self.nolabel = True
        self.id_list: list = []
        self.model = {}

        set_logging()

        self.save_dir = Path(
            increment_path(Path(self.output) / self.name, exist_ok=False)
        )
        self.tracker = Sort()
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

    def append_id(self, new_id: int) -> int:
        """Append ID value for privacy controls (specific and exclusion)

        Args:
            new_id (int): New ID
        """
        if isinstance(new_id, int):
            self.id_list.append(new_id)
        else:
            raise ValueError("Invalid ID to append, must be of type integer")

    def pop_id(self, id: int):
        """Pops specified from the ID list for privacy controls (specific and exclusion)

        Args:
            id (int): ID to remove/pop
        """
        if isinstance(id, int):
            if not self.id_list:
                self.id_list.pop(id)
        else:
            raise ValueError("Invalid ID to pop, must be of type integer")

    def set_blur_type(self, new_type: str = "default"):
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
        img = self.__to_ndarray(frame)
        img = self.__preprocess(img)
        pred = self.__inference(img)
        results = self.__postprocess(pred, img, frame)

        # [frame, bboxes] = results
        return results

    def detect_image(self, source):
        og_img = cv2.imread(source)
        img = self.__to_ndarray(og_img)
        img = self.__preprocess(img)
        pred = self.__inference(img)
        [result_img, bboxes] = self.__postprocess(pred, img, og_img)

        print(bboxes)
        cv2.imshow("Occultus", result_img)
        cv2.waitKey(0)  # 1 millisecond

    def detect_video(self, source):
        cap = cv2.VideoCapture(source)
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

    def detect_input(self, source: str = "0", frame_interval=2):
        if source.isnumeric():
            source = eval(source)

        cap = cv2.VideoCapture(source)
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
                )

                new_preds = None

                if identities is not None:
                    for index, idt in enumerate(identities):
                        new_preds = {"id": int(idt), "box": bbox_xyxy[index]}
                        bboxes.append(new_preds)

            else:
                result_img = og_img

        return [result_img, bboxes]
