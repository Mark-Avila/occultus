import customtkinter as ctk
import cv2
from PIL import Image
from occultus.core import Occultus
import threading
import os

from interface.pages.detect import *


class VideoPage(ctk.CTkToplevel):
    def __init__(self):
        ctk.CTkToplevel.__init__(self)
        self.title("Fullscreen App")
        self.state("zoomed")

        self.source = "video/mememe.mp4"
        self.filename = ""
        self.vid_cap = None
        self.current_frame = None
        self.current_frame_num = 1
        self.prev_slider_val = 0
        self.running = True
        self.num_frames = 128
        self.fps = 24
        self.slider_debounce_interval = 200  # ms
        self.slider_debounce_id = None
        self.current_progress = 0

        self.id_list = []

        self.is_playing = False
        self.is_detecting = True

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create a container frame
        self.detect_container = ctk.CTkFrame(self, fg_color="transparent")
        self.detect_container.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True)

        self.detect_page = DetectPage(self.detect_container, self)
        self.detect_page.configure(fg_color="transparent")
        self.detect_page.pack(fill="both", expand=True)

        # Start the webcam thread
        detect_thread = threading.Thread(target=self.detect_thread)
        detect_thread.start()

        self.check_isdetecting()

    def on_privacy_select(self, value: str):
        print(value.lower())

    def on_censor_select(self, value: str):
        keys = {
            "Gaussian": "gaussian",
            "Pixelized": "pixel",
            "Fill": "fill",
            "Detect": "default",
        }

        print(keys[value])

    def check_isdetecting(self):
        if self.is_detecting:
            self.after(1000, self.check_isdetecting)
        else:
            self.detect_container.destroy()

            # Create a sidebar
            self.sidebar = ctk.CTkFrame(self)
            self.sidebar.pack(side=ctk.LEFT, fill=ctk.Y)

            # Create a container frame
            self.container = ctk.CTkFrame(self, fg_color="transparent")
            self.container.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True)

            censor_label = ctk.CTkLabel(self.sidebar, text="Censor type")
            censor_label.pack(pady=5)

            censor_options = ["Gaussian", "Pixelized", "Fill", "Detect"]
            censor_privacy = ctk.CTkOptionMenu(
                master=self.sidebar,
                values=censor_options,
                command=self.on_censor_select,
                width=240,
            )

            censor_privacy.pack(padx=30, pady=5)

            privacy_label = ctk.CTkLabel(self.sidebar, text="Privacy type")
            privacy_label.pack(pady=5)

            privacy_options = ["All", "Specific", "Exclude"]

            select_privacy = ctk.CTkOptionMenu(
                master=self.sidebar,
                values=privacy_options,
                command=self.on_privacy_select,
                width=240,
            )
            select_privacy.pack(padx=30, pady=5)

            input_id_label = ctk.CTkLabel(self.sidebar, text="Add/Remove IDs")
            input_id_label.pack(padx=30, pady=5)

            self.input_id = ctk.CTkTextbox(self.sidebar, height=32)
            self.input_id.pack(padx=0, pady=5)

            btns_container = ctk.CTkFrame(
                self.sidebar, fg_color="transparent", width=150
            )
            btns_container.pack(padx=30, pady=5)

            input_add_btn = ctk.CTkButton(
                btns_container, text="Add", width=75, command=self.add_id
            )
            input_add_btn.pack(padx=2, pady=5, side=ctk.LEFT)

            input_remove_btn = ctk.CTkButton(
                btns_container, text="Remove", width=75, command=self.remove_id
            )
            input_remove_btn.pack(padx=2, pady=5, side=ctk.LEFT)

            self.id_listbox = ScrollableLabelButtonFrame(self.sidebar, height=128)
            self.id_listbox.pack(padx=30, pady=5, fill="x")

            for id in self.id_list:
                self.id_listbox.add_item(id)

            # Video feed display
            self.video_feed = ctk.CTkLabel(
                self.container,
                fg_color="#000000",
                text="Loading",
                width=720,
                height=480,
            )
            self.video_feed.pack(pady=(40, 0))

            self.video_slider = ctk.CTkSlider(
                self.container,
                width=720,
                from_=0,
                to=self.num_frames,
                command=self.on_slider_change,
            )
            self.video_slider.pack(pady=(40, 0))
            self.video_slider.set(self.current_frame_num)

            player_wrapper = ctk.CTkFrame(
                self.container, width=720, fg_color="transparent"
            )
            player_wrapper.pack(pady=(40, 0))

            self.video_start_btn = ctk.CTkButton(player_wrapper, width=80, text="Start")
            self.video_start_btn.pack(padx=20, side=ctk.LEFT)
            self.video_play_btn = ctk.CTkButton(
                player_wrapper, width=80, text="Play", command=self.on_video_play
            )
            self.video_play_btn.pack(padx=20, side=ctk.LEFT)
            self.video_end_btn = ctk.CTkButton(player_wrapper, width=80, text="End")
            self.video_end_btn.pack(padx=20, side=ctk.LEFT)

            self.video_feed.bind("<Button-1>", self.on_feed_click)

            # Start the webcam thread
            video_thread = threading.Thread(target=self.video_thread)
            video_thread.start()

            self.update_feed()

            # self.edit_page = EditPage(self.container, self)
            # self.edit_page.pack(fill="both", expand=True)

    def update_feed(self):
        self.after(int(1000 / self.fps), self.update_feed)

        # Update the Tkinter GUI with the latest frame
        if hasattr(self, "current_frame"):
            # if self.is_recording:
            #     self.occultus.save_video(frame=self.raw_frame, iterables=self.iterables)
            self.video_feed.configure(text="", image=self.current_frame)

        if self.is_playing:
            self.video_slider.set(self.current_frame_num)

    def on_slider_change(self, value):
        # Cancel the previous after event if it exists
        if self.slider_debounce_id is not None:
            self.after_cancel(self.slider_debounce_id)

        # Schedule a new after event
        self.slider_debounce_id = self.after(
            self.slider_debounce_interval, self.debounced_on_slider_change, value
        )

    def debounced_on_slider_change(self, value):
        # This function will be called when the timer times out
        # Implement the actual logic you want to perform
        value = int(value)
        if value != self.prev_slider_val:
            self.prev_slider_val = value
            self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, value - 1)
            _, frame = self.vid_cap.read()
            frame = self.__imread_to_ctk(frame)
            self.current_frame_num = value
            self.current_frame = frame

    def on_video_play(self):
        self.video_play_btn.configure(text="Pause", command=self.on_video_pause)
        self.video_slider.configure(state="disabled")
        self.is_playing = True

    def on_video_pause(self):
        self.video_play_btn.configure(text="Play", command=self.on_video_play)
        self.video_slider.configure(state="normal")
        self.is_playing = False

    def update_slider_numframes(self):
        self.video_slider.configure(to=self.num_frames)

    def update_slider_currframe(self):
        self.video_slider.configure(to=self.num_frames)

    def add_id(self):
        new_id = self.input_id.get("1.0", "end-1c")

        if new_id.isdigit():
            self.id_list.append(new_id)
            self.id_listbox.add_item(new_id)

        if new_id:
            self.input_id.delete("1.0", "end")

    def remove_id(self):
        id = self.input_id.get("1.0", "end-1c")

        if id.isdigit():
            self.id_list.remove(id)
            self.id_listbox.remove(id)

        if id:
            self.input_id.delete("1.0", "end")

    def video_thread(self):
        self.vid_cap = cv2.VideoCapture(f"cache/{self.filename}")

        if not self.vid_cap.isOpened():
            self.vid_cap.release()
            raise Exception("Failed to load video")

        self.num_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

        self.after(0, self.update_slider_numframes)

        _, img = self.vid_cap.read()
        self.current_frame = self.__imread_to_ctk(img)

        while True:
            if not self.running:
                break

            if self.is_playing:
                ret, og_img = self.vid_cap.read()

                if not ret:
                    self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_num = 0

                    _, og_img = self.vid_cap.read()
                    imgtk = self.__imread_to_ctk(og_img)
                    self.current_frame = imgtk

                    self.is_playing = False

                if ret:
                    imgtk = self.__imread_to_ctk(og_img)
                    self.current_frame_num = self.vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
                    self.current_frame = imgtk

            # Introduce a delay to match the video frame rate (assuming 30 frames per second)
            delay = int(1000 / 30)  # Delay to achieve 30 frames per second
            cv2.waitKey(delay)

    def update_progress(self):
        if self.is_detecting:
            self.after(100, self.update_progress)

        # Update the Tkinter GUI with the latest frame
        if hasattr(self, "current_progress"):
            self.detect_page.set_progress(self.current_progress)

    def detect_thread(self):
        self.filename = os.path.basename(self.source)

        occultus = Occultus(
            weights="weights/kamukha-v3.pt",
            conf_thres=0.25,
            output_folder="cache",
            output_name=self.filename,
            output_create_folder=False,
            show_label=True,
        )
        occultus.set_blur_type("default")

        self.is_detecting = True
        self.after(0, self.update_progress)

        for _, frame_num, max_frames in occultus.detect_video_generator(self.source):
            progress_value = frame_num / max_frames
            self.current_progress = progress_value

        self.is_detecting = False

    def __imread_to_ctk(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(img)
        imgtk = ctk.CTkImage(img, size=(720, 480))
        return imgtk

    def on_close(self):
        # Release the video feed and close the window
        self.running = False

        if self.vid_cap is not None and self.vid_cap.isOpened():
            self.vid_cap.release()

        self.destroy()

    def on_feed_click(self, event):
        # Get the coordinates of the mouse click relative to the label
        coords = (event.x, event.y)

        print(coords)


class ScrollableLabelButtonFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.command = command
        self.radiobutton_variable = ctk.StringVar()
        self.label_list = []
        self.button_list = []

    def add_item(self, item, image=None):
        label = ctk.CTkLabel(
            self, text=item, image=image, compound="left", padx=5, anchor="w"
        )
        button = ctk.CTkButton(self, text="Remove", width=100, height=24)
        if self.command is not None:
            button.configure(command=lambda: self.command(item))
        label.grid(row=len(self.label_list), column=0, pady=(0, 10), sticky="w")
        button.grid(row=len(self.button_list), column=1, pady=(0, 10), padx=5)
        self.label_list.append(label)
        self.button_list.append(button)

    def remove_item(self, item):
        for label, button in zip(self.label_list, self.button_list):
            if item == label.cget("text"):
                label.destroy()
                button.destroy()
                self.label_list.remove(label)
                self.button_list.remove(button)
                return
