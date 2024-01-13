import customtkinter as ctk
import cv2
from PIL import Image
from occultus.core import Occultus
import threading
import os
import subprocess
import shutil

from interface.pages.detect import *


class VideoPage(ctk.CTkToplevel):
    def __init__(self, controller, source):
        ctk.CTkToplevel.__init__(self)
        self.title("Occultus Video")
        self.state("zoomed")

        self.controller = controller
        self.source = source

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
        self.privacy_mode = "all"
        self.censor_mode = "default"

        self.is_playing = False
        self.is_detecting = True

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create a container frame
        self.detect_container = ctk.CTkFrame(self, fg_color="transparent")
        self.detect_container.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True)

        self.detect_page = DetectPage(self.detect_container, self)
        self.detect_page.configure(fg_color="transparent")
        self.detect_page.pack(fill="both", expand=True)

        # # Start the webcam thread
        detect_thread = threading.Thread(
            target=self.detect_thread,
            args=("weights/kamukha-v3.pt", "cache", self.filename, False, False),
        )
        # Start the webcam thread
        detect_thread.start()

        self.check_isdetecting()

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

            self.input_id = ctk.CTkEntry(self.sidebar, height=32)
            self.input_id.pack(padx=0, pady=5)

            btns_container = ctk.CTkFrame(
                self.sidebar, fg_color="transparent", width=150
            )
            btns_container.pack(padx=30, pady=5)

            input_add_btn = ctk.CTkButton(
                btns_container, text="Add", width=75, command=self.add_id
            )
            input_add_btn.pack(padx=2, pady=5, side=ctk.LEFT)

            self.id_listbox = ScrollableLabelButtonFrame(
                self.sidebar, height=128, command=self.remove_id_click
            )
            self.id_listbox.pack(padx=30, pady=5, fill="x")

            render_button = ctk.CTkButton(
                self.sidebar, text="Render", fg_color="red", command=self.on_render
            )
            render_button.pack(padx=30, pady=5, fill="x")

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

            start_image = Image.open("assets/start.png")
            self.video_start_btn = ctk.CTkButton(
                player_wrapper,
                width=80,
                height=80,
                text="",
                image=ctk.CTkImage(start_image, size=(40, 40)),
                fg_color="transparent",
            )
            self.video_start_btn.pack(padx=20, side=ctk.LEFT)

            play_image = Image.open("assets/play.png")
            self.video_play_btn = ctk.CTkButton(
                player_wrapper,
                width=80,
                height=80,
                text="",
                command=self.on_video_play,
                image=ctk.CTkImage(play_image, size=(40, 40)),
                fg_color="transparent",
            )
            self.video_play_btn.pack(padx=20, side=ctk.LEFT)

            end_image = Image.open("assets/end.png")
            self.video_end_btn = ctk.CTkButton(
                player_wrapper,
                width=80,
                height=80,
                text="",
                image=ctk.CTkImage(end_image, size=(40, 40)),
                fg_color="transparent",
            )
            self.video_end_btn.pack(padx=20, side=ctk.LEFT)

            self.video_feed.bind("<Button-1>", self.on_feed_click)

            # Start the webcam thread
            video_thread = threading.Thread(target=self.video_thread)
            video_thread.start()

            self.update_feed()

            # self.edit_page = EditPage(self.container, self)
            # self.edit_page.pack(fill="both", expand=True)

    def update_feed(self):
        if self.running:
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
        pause_image = Image.open("assets/pause.png")
        self.video_play_btn.configure(
            command=self.on_video_pause,
            image=ctk.CTkImage(pause_image, size=(40, 40)),
        )
        self.video_slider.configure(state="disabled")
        self.is_playing = True

    def on_video_pause(self):
        play_image = Image.open("assets/play.png")
        self.video_play_btn.configure(
            command=self.on_video_play,
            image=ctk.CTkImage(play_image, size=(40, 40)),
        )
        self.video_slider.configure(state="normal")
        self.is_playing = False

    def update_slider_numframes(self):
        self.video_slider.configure(to=self.num_frames)

    def update_slider_currframe(self):
        self.video_slider.configure(to=self.num_frames)

    def add_id(self):
        new_id = self.input_id.get()

        if new_id.isdigit():
            self.id_list.append(int(new_id))
            self.id_listbox.add_item(new_id)

        if new_id:
            self.input_id.delete(0, ctk.END)

    def remove_id_click(self, item):
        if item.isdigit():
            self.id_list.remove(int(item))
            self.id_listbox.remove_item(item)

        if item:
            self.input_id.delete(0, ctk.END)

    def on_privacy_select(self, value: str):
        value = value.lower()
        self.privacy_mode = value

    def on_censor_select(self, value: str):
        keys = {
            "Gaussian": "gaussian",
            "Pixelized": "pixel",
            "Fill": "fill",
            "Detect": "default",
        }

        self.censor_mode = keys[value]

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

        while self.running:
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

    def detect_thread(
        self,
        weights: str,
        output: str,
        output_name: str,
        output_create_folder: bool,
        show_label: bool,
        blur_type: str = "default",
        select_type: str = "all",
        id_list=[],
    ):
        self.filename = os.path.basename(self.source)

        occultus = Occultus(
            weights=weights,
            conf_thres=0.25,
            output_folder=output,
            output_name=output_name,
            output_create_folder=output_create_folder,
            show_label=show_label,
        )

        occultus.set_blur_type(
            blur_type, show_label=True if blur_type == "default" else False
        )
        occultus.set_privacy_control(select_type)

        for id in id_list:
            occultus.add_id(id)

        self.is_detecting = True
        self.after(0, self.update_progress)

        for _, frame_num, max_frames in occultus.detect_video_generator(self.source):
            progress_value = frame_num / max_frames
            self.current_progress = progress_value

        self.is_detecting = False

    def render_check_isdetecting(self):
        if self.is_detecting:
            self.after(1000, self.render_check_isdetecting)
        else:
            self.detect_container.destroy()

            self.rowconfigure(0, weight=1)
            self.columnconfigure(0, weight=1)
            self.rowconfigure(2, weight=1)
            self.columnconfigure(2, weight=1)

            wrapper = ctk.CTkFrame(self, fg_color="transparent")
            wrapper.grid(row=1, column=1)

            done_label = ctk.CTkLabel(
                wrapper, text="Render done!", font=ctk.CTkFont(size=24, weight="bold")
            )
            done_label.pack(pady=10)
            check_label = ctk.CTkLabel(wrapper, text="Check on output folder")
            check_label.pack(pady=10)

            check_button = ctk.CTkButton(
                wrapper,
                text="Open Output folder",
                height=40,
                width=256,
                command=self.on_output_check,
            )
            check_button.pack(pady=10)

    def on_output_check(self):
        subprocess.run(["explorer", "output"])

    def on_render(self):
        self.running = False

        while self.running:
            if not self.running:
                break

        self.container.destroy()
        self.sidebar.destroy()

        # Create a container frame
        self.detect_container = ctk.CTkFrame(self, fg_color="transparent")
        self.detect_container.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True)

        self.detect_page = DetectPage(
            self.detect_container, self, text="Rendering Video"
        )
        self.detect_page.configure(fg_color="transparent")
        self.detect_page.pack(fill="both", expand=True)

        self.detect_page.set_progress(0)

        new_idlist = []
        for id in self.id_list:
            new_idlist.append(id)

        # Start the webcam thread
        detect_thread = threading.Thread(
            target=self.detect_thread,
            args=("weights/kamukha-v3.pt", "output", self.filename, False, False),
            kwargs={
                "blur_type": self.censor_mode,
                "select_type": self.privacy_mode,
                "id_list": new_idlist,
            },
        )
        detect_thread.start()

        self.is_detecting = True
        self.render_check_isdetecting()

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

        try:
            shutil.rmtree("cache")
        except OSError as e:
            Exception(e)

        # Close main application window
        self.controller.on_close()

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
