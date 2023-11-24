import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

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
from occultus.utils.plots import (
    draw_boxes,
)
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
    def __init__(self, weights):
        self.weights = weights
        self.conf_thres = 0.25
        self.iou = 0.45
        self.device = ""
        self.nosave = False
        self.output = "output/detect"
        self.name = "inf"
        self.track = False
        self.show_fps = False
        self.thickness = 2
        self.nobbox = False
        self.nolabel = False
        self.view_image = True
        self.id_list: list = []
        self.view_image_test = None
        self.flipped = False
        self.model = {}
        pass

    def append_id(self, new_id):
        self.id_list.append(new_id)

    def pop_id(self, curr_id):
        if not self.id_list:
            self.id_list.pop(curr_id)

    def onMouse(event, x, y, flags, param):
        global posList
        if event == cv2.EVENT_LBUTTONDOWN:
            posList.append((x, y))

    def load_video(self, source: str, img_size=640):
        self.source = source
        self.img_size = img_size
        self.view_image = False
        self.nosave = False

    def load_stream(self, source: str = "0", img_size=640, flipped=False):
        self.source = source
        self.img_size = img_size
        self.view_image = True
        self.flipped = flipped
        self.nosave = True

    def set_config(self, config: dict):
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

    def inference(self, dataset):
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
        names = (
            self.model["model"].names
            if hasattr(self.model["model"], "module")
            else self.model["model"].names
        )
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        im0 = None

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

                im0 = draw_boxes(
                    im0,
                    bbox_xyxy,
                    identities=identities,
                    categories=categories,
                    confidences=confidences,
                    names=names,
                    colors=colors,
                    nobbox=self.nobbox,
                    nolabel=self.nolabel,
                    thickness=self.thickness,
                    id_list=self.id_list,
                )

        return im0

    def show_frame(frame):
        cv2.imshow("Face", frame)
        cv2.waitKey(1)  # 1 millisecond

    def save_img(self, frame):
        cv2.imwrite(self.model["save_path"], frame)
        print(f"The image with the result is saved in: {self.model['save_path']}")

    def save_video(self, frame, iterables):
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
                        identities=identities,
                        categories=categories,
                        confidences=confidences,
                        names=names,
                        colors=colors,
                        nobbox=self.nobbox,
                        nolabel=self.nolabel,
                        thickness=self.thickness,
                        id_list=self.id_list,
                        flipped=self.flipped,
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
