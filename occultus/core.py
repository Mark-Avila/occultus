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
from occultus.utils.datasets import LoadStreams, LoadImages
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
    def __init__(self, weights: str):
        """Occultus Constructor

        Args:
            weights (str): Path of weights (pt) for face detection

        Example:
        ```
        from occultus.core import Occultus
        occultus = Occultus("path/to/weights")
        ```
        """
        self.weights = weights
        self.conf_thres = 0.25
        self.iou = 0.45
        self.device = ""
        self.nosave = False
        self.output = "output"
        self.name = datetime.datetime.now().strftime("%b_%d_%Y-%H_%M_%S_")
        self.track = False
        self.show_fps = False
        self.thickness = 2
        self.nobbox = False
        self.nolabel = True
        self.view_image = True
        self.id_list: list = []
        self.view_image_test = None
        self.flipped = False
        self.blur_type = "detect"
        self.select_type = "all"
        self.model = {}
        pass

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

    def load_video(self, source: str, img_size: int = 640):
        """
        Load a video file for face detection and blurring.

        Args:
            source (str): The path to the video file.
            img_size (int, optional): The size of each frame for model input. Defaults to 640.

        Raises:
            FileNotFoundError: If the specified video file is not found.
            ValueError: If the provided video format is not supported.

        Notes:
            - This method initializes the video loading process for subsequent face detection and blurring.
            - Supported video formats are [.mov, .avi, .mp4, .mpg, .mpeg, .m4v, .wmv, .mkv].
            - The default image size for model input is set to 640 pixels.

        Usage:
            To load a video, use the following example:
            ```python
            instance_name.load_video("path/to/video.mp4", img_size=720)
            ```
        """

        if isinstance(source, str) is False:
            raise ValueError("Invalid source value, must be of type string")
        if isinstance(img_size, int) is False:
            raise ValueError("Invalid img_size value, must be of type integer")

        self.source = source
        self.img_size = img_size
        self.view_image = False
        self.nosave = False

    def load_stream(self, source: str = "0", img_size=640, flipped=False):
        """
        Load a video stream or live feed for face detection and blurring.

        Args:
            source (str, optional):
            - For video streams: The URL or IP address of the video stream.
            - For live feed: The camera index (usually 0 for default camera). Defaults to "0".\n
            img_size (int, optional): The size of each frame for model input. Defaults to 640.
            flipped (bool, optional): Flag to enable/disable horizontal flipping of each frame. Defaults to False.

        Raises:
            ValueError: If the provided video stream source is invalid or if the camera index is invalid.

        Notes:
            - This method initializes the video stream loading process for subsequent face detection and blurring.
            - If using a video stream, provide the URL or IP address as the 'source' parameter.
            - If using a live feed, provide the camera index as the 'source' parameter.
            - The default image size for model input is set to 640 pixels.

        Usage:
            To load a video stream, use the following example:
            ```python
            instance_name.load_stream("http://example.com/stream", img_size=720, flipped=True)
            ```

            To load a live feed from the default camera, use the following example:
            ```python
            instance_name.load_stream("0", img_size=720, flipped=True)
            ```
        """

        if isinstance(source, str) is False:
            raise ValueError("Invalid source value, must be of type string")
        if isinstance(img_size, int) is False:
            raise ValueError("Invalid img_size value, must be of type integer")
        if isinstance(flipped, bool) is False:
            raise ValueError("Invalid flipped value, must be of type boolean")

        self.source = source
        self.img_size = img_size
        self.view_image = True
        self.flipped = flipped
        self.nosave = True

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
                "Invalid censor type. Please choose between the following: [default, gaussian, pixel, fill]"
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
                "Invalid select mode. Please choose between the following: ['all', 'specific', 'exclude']"
            )

        return

    def set_config(self, config: dict):
        """
        Set configuration parameters based on the values provided in the input dictionary.

        Parameters:
        - config (dict): A dictionary containing configuration parameter values.

        Default values:
        ```python
        config = {
            "conf-thres" (float): 0.25,
            "iou" (float): 0.45,
            "device" (str): "",
            "nosave" (bool): False,
            "output" (str): "output",
            "name" (str): "vid",
            "track" (bool): False,
            "show-fps" (bool): False,
            "thickness" (int): 2,
            "nobbox" (bool): False,
            "nolabel" (bool): True,
            "flipped" (bool): False
        }

        instance_of_your_class.set_config(config)
        ```

        Note:
        - If a parameter is not present in the input dictionary, the default value is retained.
        """
        self.conf_thres = config.get("conf-thres", self.conf_thres)
        self.iou = config.get("iou", self.iou)
        self.device = config.get("device", self.device)
        self.nosave = config.get("nosave", self.nosave)
        self.output = config.get("output", self.output)
        self.name = config.get("name", self.name)
        self.track = config.get("track", self.track)
        self.show_fps = config.get("show-fps", self.show_fps)
        self.thickness = config.get("thickness", self.thickness)
        self.nobbox = config.get("nobbox", self.nobbox)
        self.nolabel = config.get("nolabel", self.nolabel)
        self.flipped = config.get("flipped", self.flipped)

    def initialize(self):
        """
        Initialize the object detection system.

        This method performs various initialization tasks, including setting up configuration parameters,
        loading the detection model, configuring the data loader, and preparing the output directory.

        Example:
        ```python
        occultus = Occultus("path/to/weights")

        frames = occultus.initialize() # Must be called before inference, process, save video methods

        for pred, dataset, iterables in occultus.inference(frames):
            processed_preds = occultus.process(pred, dataset, iterables)
            occultus.save_video(frame, iterables)

        #... rest of the code
        ```

        Returns:
        - dataset: The initialized dataset for processing video frames.

        Note:
        - This method is MUST be called once before starting the object detection process.
        """
        trace = False
        self.model["augment"] = False
        self.model["sort_tracker"] = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

        self.save_img = not self.nosave and not self.source.endswith(".txt")
        self.model["webcam"] = (
            self.source.isnumeric()
            or self.source.endswith(".txt")
            or self.source.lower().startswith(
                ("rtsp://", "rtmp://", "http://", "https://")
            )
        )
        self.model["save_dir"] = Path(
            increment_path(Path(self.output) / self.name, exist_ok=False)
        )  # increment run

        if not self.nosave:
            (self.model["save_dir"]).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.model["device"] = select_device(self.device)
        self.model["half"] = (
            self.model["device"].type != "cpu"
        )  # half precision only supported on CUDA

        # Load model
        self.model["model"] = attempt_load(
            self.weights, map_location=self.model["device"]
        )  # load FP32 model
        stride = int(self.model["model"].stride.max())  # model stride
        self.model["imgsz"] = check_img_size(self.img_size, s=stride)  # check img_size

        if trace:
            self.model["model"] = TracedModel(
                self.model["model"], self.model["device"], self.img_size
            )

        if self.model["half"]:
            self.model["model"].half()  # to FP16

        # Second-stage classifier
        self.model["classify"] = False
        if self.model["classify"]:
            self.model["modelc"] = load_classifier(name="resnet101", n=2)  # initialize
            self.model["modelc"].load_state_dict(
                torch.load("weights/resnet101.pt", map_location=self.model["device"])[
                    "model"
                ]
            ).to(self.model["device"]).eval()

        # Set Dataloader
        self.model["vid_path"], self.model["vid_writer"] = None, None
        if self.model["webcam"]:
            self.view_image = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(
                self.source, img_size=self.model["imgsz"], stride=stride
            )
        else:
            dataset = LoadImages(
                self.source, img_size=self.model["imgsz"], stride=stride
            )

        # Run inference once
        if self.model["device"].type != "cpu":
            self.model["model"](
                torch.zeros(1, 3, self.model["imgsz"], self.model["imgsz"])
                .to(self.model["device"])
                .type_as(next(self.model["model"].parameters()))
            )  # run once

        return dataset

    def inference(self, dataset):
        """
        Perform object detection inference on the given dataset.

        This method iterates over the frames in the provided dataset, processes each frame using the
        pre-loaded detection model, and yields the predictions along with additional information for each frame.

        Parameters:
        - dataset: The dataset containing frames for object detection.

        Yields:
        - pred: List of dictionaries containing bounding box predictions for detected objects.
        - dataset: The original dataset.
        - iterables: Dictionary containing additional information for the current frame, crucial for using the ```process``` method, including:
            - "path": Path to the image or video file.
            - "im0s": Original image.
            - "img": Processed image tensor.
            - "vid_cap": Video capture object.

        Example:
        ```python
        occultus = Occultus("path/to/weights")

        frames = occultus.initialize()

        for pred, dataset, iterables in frames.inference(frames):
            processed_preds = frames.process(pred, dataset, iterables)
            frames.save_video(frame, iterables)

        #... rest of the code
        ```

        Note:
        - The `inference` method uses the pre-loaded model from ```initialize``` method to perform object detection on each frame
        in the dataset. It yields the predictions, the original dataset, and additional information
        for further processing.
        """
        old_img_w = old_img_h = self.model["imgsz"]
        old_img_b = 1
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.model["device"])
            img = img.half() if self.model["half"] else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.model["device"].type != "cpu" and (
                old_img_b != img.shape[0]
                or old_img_h != img.shape[2]
                or old_img_w != img.shape[3]
            ):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model["model"](img, augment=self.model["augment"])[0]

            # Inference
            # t1 = time_synchronized()
            pred = self.model["model"](img, augment=self.model["augment"])[0]
            # t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou,
                classes=0,
                agnostic=False,
            )
            # t3 = time_synchronized()

            # Apply Classifier
            if self.model["classify"]:
                pred = apply_classifier(pred, self.model["modelc"], img, im0s)

            iterables = {"path": path, "im0s": im0s, "img": img, "vid_cap": vid_cap}

            yield pred, dataset, iterables

    def process(self, pred, dataset, iterables):
        """
        Process object detection predictions and generate visualizations.

        This method takes the predictions from the object detection model, processes the detections,
        tracks objects using the SORT tracker, applies blurring or other visual effects based on the
        configuration, and returns the processed image along with information about detected objects.

        Parameters:
        - pred: List of dictionaries containing bounding box predictions for detected objects.
        - dataset: The original dataset.
        - iterables: Dictionary containing additional information for the current frame, including:
            - "path": Path to the image or video file.
            - "im0s": Original image.
            - "img": Processed image tensor.
            - "vid_cap": Video capture object.

        Returns:
        - result: A list containing the processed image (numpy array) and a list of dictionaries,
        each containing information about a detected object, including its identity and bounding box.

        Example:
        ```python
        occultus = Occultus("path/to/weights")

        frames = occultus.initialize()

        for pred, dataset, iterables in occultus.inference(frames):
            processed_preds = occultus.process(pred, dataset, iterables) //OR
            [frame, dets] = occultus.process(pred, dataset, iterables) //OR

            occultus.save_video(frame, iterables) # For video
            occultus.show_frame(frame) # For streams

        #... rest of the code
        ```

        Note:
        - The `process` method applies tracking, visualization, and blurring effects based on the
        configuration settings. It returns the processed image and a list of dictionaries containing
        information about detected objects, including their identities and bounding boxes.
        """
        names = (
            self.model["model"].names
            if hasattr(self.model["model"], "module")
            else self.model["model"].names
        )

        im0 = None
        bboxes = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if self.model["webcam"]:  # batch_size >= 1
                p, s, im0, frame = (
                    iterables["path"][i],
                    "%g: " % i,
                    iterables["im0s"][i].copy(),
                    dataset.count,
                )
            else:
                p, s, im0, frame = (
                    iterables["path"],
                    "",
                    iterables["im0s"],
                    getattr(dataset, "frame", 0),
                )

            p = Path(p)  # to Path
            self.model["save_path"] = str(self.model["save_dir"] / p.name)  # img.jpg
            self.model["txt_path"] = str(self.model["save_dir"] / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    iterables["img"].shape[2:], det[:, :4], im0.shape
                ).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0, 6))
                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack(
                        (dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass]))
                    )

                tracked_dets = self.model["sort_tracker"].update(dets_to_sort, False)
                tracks = self.model["sort_tracker"].getTrackers()

                # draw boxes for visualization
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    confidences = None

                    if self.track:
                        # loop over tracks
                        for t, track in enumerate(tracks):
                            track_color = (0, 0, 255)

                            [
                                cv2.line(
                                    im0,
                                    (
                                        int(track.centroidarr[i][0]),
                                        int(track.centroidarr[i][1]),
                                    ),
                                    (
                                        int(track.centroidarr[i + 1][0]),
                                        int(track.centroidarr[i + 1][1]),
                                    ),
                                    track_color,
                                    thickness=self.thickness,
                                )
                                for i, _ in enumerate(track.centroidarr)
                                if i < len(track.centroidarr) - 1
                            ]
                else:
                    bbox_xyxy = dets_to_sort[:, :4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]

                blur_functions = {
                    "gaussian": blur_boxes,
                    "pixel": pixelate_boxes,
                    "fill": fill_boxes,
                    "detect": draw_boxes,
                }

                blur_function = blur_functions.get(self.blur_type, draw_boxes)

                im0 = blur_function(
                    im0,
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

        return [im0, bboxes]

    def show_frame(self, frame):
        """
        Display the given frame in a window titled "Face".

        Parameters:
        - frame: The frame (image) to be displayed.

        Example:
        ```python
        instance_of_your_class.show_frame(frame)
        ```

        Note:
        - This method uses OpenCV to show the provided frame in a window titled "Face".
        - The window is updated every millisecond.
        """
        cv2.imshow("Face", frame)
        cv2.waitKey(1)  # 1 millisecond

    def save_img(self, frame):
        """
        Save the given frame as an image file.

        Parameters:
        - frame: The frame (image) to be saved.

        Example:
        ```python
        instance_of_your_class.save_img(frame)
        ```

        Note:
        - This method saves the provided frame as an image file in the directory specified by the
        model's `save_path` attribute.
        - The file format is determined by the file extension in the `save_path`.
        - A message indicating the saved path is printed to the console.
        """
        cv2.imwrite(self.model["save_path"], frame)
        print(f"The image with the result is saved in: {self.model['save_path']}")

    def save_video(self, frame, iterables):
        """
        Save the given frame as part of a video file.

        Parameters:
        - frame: The frame (image) to be included in the video.
        - iterables: A dictionary containing iterable items such as the video capture object.

        Example:
        ```python
        instance_of_your_class.save_video(frame, iterables)
        ```

        Note:
        - This method appends the provided frame to a video file specified by the model's
        `save_path` attribute.
        - If the video file is a new one, it initializes a new video writer and releases the
        previous one if it exists.
        - The file format is determined by the file extension in the `save_path`.
        - The video's frames per second (fps), width, and height are determined based on the video
        capture object or default values for a stream.
        - If the video writer is not initialized, this method creates a new video writer and writes
        the frame to the video file.
        """
        if self.model["vid_path"] != self.model["save_path"]:  # new video
            self.model["vid_path"] = self.model["save_path"]
            if isinstance(self.model["vid_writer"], cv2.VideoWriter):
                self.model["vid_writer"].release()  # release previous video writer
            if iterables["vid_cap"]:  # video
                fps = iterables["vid_cap"].get(cv2.CAP_PROP_FPS)
                w = int(iterables["vid_cap"].get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(iterables["vid_cap"].get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:  # stream
                fps, w, h = 30, frame.shape[1], frame.shape[0]
                self.model["save_path"] += ".mp4"
            self.model["vid_writer"] = cv2.VideoWriter(
                self.model["save_path"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        self.model["vid_writer"].write(frame)

    def run(self, log=True, ext_frame=None):
        """
        Run object detection on the specified source (image, video, or webcam stream).

        This method initializes the object detection model, performs inference on each frame
        of the provided source, applies tracking, visualization, and blurring effects, and optionally
        saves the results. The processed frames are displayed if the `view_image` option is enabled.

        Parameters:
        - log (bool): Whether to enable logging during the detection process.
        - ext_frame: An optional external frame for displaying the processed frames.

        Example:
        ```python
        instance_of_your_class.run(log=True, ext_frame=external_frame)
        ```

        Note:
        - The `run` method sets up the necessary components, including the model, dataloader, and
        visualization settings. It then processes each frame of the specified source, applies
        object detection, tracking, and visualization, and optionally saves the results.
        """
        trace = False
        augment = False
        sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

        save_img = not self.nosave and not self.source.endswith(
            ".txt"
        )  # save inference images
        webcam = (
            self.source.isnumeric()
            or self.source.endswith(".txt")
            or self.source.lower().startswith(
                ("rtsp://", "rtmp://", "http://", "https://")
            )
        )
        save_dir = Path(
            increment_path(Path(self.output) / self.name, exist_ok=False)
        )  # increment run
        if not self.nosave:
            (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, self.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name="resnet101", n=2)  # initialize
            modelc.load_state_dict(
                torch.load("weights/resnet101.pt", map_location=device)["model"]
            ).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            self.view_image = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, "module") else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != "cpu":
            model(
                torch.zeros(1, 3, imgsz, imgsz)
                .to(device)
                .type_as(next(model.parameters()))
            )  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        ###################################
        startTime = 0
        ###################################

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != "cpu" and (
                old_img_b != img.shape[0]
                or old_img_h != img.shape[2]
                or old_img_w != img.shape[3]
            ):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou,
                classes=0,
                agnostic=False,
            )
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = (
                        path[i],
                        "%g: " % i,
                        im0s[i].copy(),
                        dataset.count,
                    )
                else:
                    p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / "labels" / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}"
                )  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    dets_to_sort = np.empty((0, 6))
                    # NOTE: We send in detected object class too
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack(
                            (dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass]))
                        )

                    tracked_dets = sort_tracker.update(dets_to_sort, False)
                    tracks = sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        if self.track:
                            # loop over tracks
                            for t, track in enumerate(tracks):
                                track_color = (0, 0, 255)

                                [
                                    cv2.line(
                                        im0,
                                        (
                                            int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1]),
                                        ),
                                        (
                                            int(track.centroidarr[i + 1][0]),
                                            int(track.centroidarr[i + 1][1]),
                                        ),
                                        track_color,
                                        thickness=self.thickness,
                                    )
                                    for i, _ in enumerate(track.centroidarr)
                                    if i < len(track.centroidarr) - 1
                                ]
                    else:
                        bbox_xyxy = dets_to_sort[:, :4]
                        identities = None
                        categories = dets_to_sort[:, 5]
                        confidences = dets_to_sort[:, 4]

                    im0 = draw_boxes(
                        im0,
                        bbox_xyxy,
                        ids=identities,
                        ids_list=self.id_list,
                        privacy=self.select_type,
                        categories=categories,
                        confidences=confidences,
                        names=names,
                        nobbox=self.nobbox,
                        nolabel=self.nolabel,
                        id_list=self.id_list,
                    )

                print(f"Done. ({(1E3 * (t2 - t1)):.1f}ms)")

                if dataset.mode != "image" and self.show_fps:
                    currentTime = time.time()

                    fps = 1 / (currentTime - startTime)
                    startTime = currentTime
                    cv2.putText(
                        im0,
                        "FPS: " + str(int(fps)),
                        (20, 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 255, 0),
                        2,
                    )

                # if ext_frame:
                #     imgtk = ImageTk.PhotoImage(image=im0)
                #     ext_frame.imgtk = imgtk
                #     ext_frame.configure(image=imgtk)

                elif self.view_image:
                    im0 = cv2.flip(im0, 1) if self.flipped else im0
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += ".mp4"
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                            )
                        vid_writer.write(im0)

        print(f"Done. ({time.time() - t0:.3f}s)")
