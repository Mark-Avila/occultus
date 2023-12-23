import customtkinter as ctk
import cv2
from PIL import Image
from occultus.core import Occultus
import threading

from interface.pages.detect import *


class VideoPage(ctk.CTkToplevel):
    def __init__(self):
        ctk.CTkToplevel.__init__(self)
        self.title("Fullscreen App")
        self.state("zoomed")

        self.vid_cap = None
        self.current_frame = None
        self.current_frame_num = 1
        self.prev_slider_val = 0
        self.is_playing = False
        self.running = True
        self.num_frames = 128
        self.fps = 24
        self.slider_debounce_interval = 200  # ms
        self.slider_debounce_id = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)

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

        detect_page = DetectPage(self.container, self)
        detect_page.configure(fg_color="transparent")
        detect_page.pack(fill="both", expand=True)

        # Video feed display
        # self.video_feed = ctk.CTkLabel(
        #     self.container, fg_color="#000000", text="Loading", width=720, height=480
        # )
        # self.video_feed.pack(pady=(40, 0))

        # self.video_slider = ctk.CTkSlider(
        #     self.container,
        #     width=720,
        #     from_=0,
        #     to=self.num_frames,
        #     command=self.on_slider_change,
        # )
        # self.video_slider.pack(pady=(40, 0))
        # self.video_slider.set(self.current_frame_num)

        # player_wrapper = ctk.CTkFrame(self.container, width=720, fg_color="transparent")
        # player_wrapper.pack(pady=(40, 0))

        # self.video_start_btn = ctk.CTkButton(player_wrapper, width=80, text="Start")
        # self.video_start_btn.pack(padx=20, side=ctk.LEFT)
        # self.video_play_btn = ctk.CTkButton(
        #     player_wrapper, width=80, text="Play", command=self.on_video_play
        # )
        # self.video_play_btn.pack(padx=20, side=ctk.LEFT)
        # self.video_end_btn = ctk.CTkButton(player_wrapper, width=80, text="End")
        # self.video_end_btn.pack(padx=20, side=ctk.LEFT)

        # self.video_feed.bind("<Button-1>", self.on_feed_click)

        # Start the webcam thread
        # video_thread = threading.Thread(target=self.video_thread)
        # video_thread.start()

        # self.update_feed()

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

    def video_thread(self):
        self.vid_cap = cv2.VideoCapture("video/crowd.mp4")

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
