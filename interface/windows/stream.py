import customtkinter as ctk
import cv2
from PIL import Image
from occultus.core import Occultus
import threading
import os
import subprocess


class StreamPage(ctk.CTkToplevel):
    def __init__(self, controller):
        ctk.CTkToplevel.__init__(self)
        self.title("Occultus Camera")
        self.controller = controller

        # Create a frame to center the content
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True)

        self.vid = None
        self.running = True
        self.is_recording = False
        self.iterables = None

        # On close handling
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Main content (Video feed wrapper)
        self.content = ctk.CTkFrame(container)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)

        # Sidebar content
        self.sidebar = ctk.CTkFrame(container, width=200)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.thread_started = False

        # Video feed display
        self.feed = ctk.CTkLabel(
            self.content, fg_color="#000000", text="Loading", width=640, height=480
        )
        self.feed.pack()

        censor_label = ctk.CTkLabel(self.sidebar, text="Censor type")
        censor_label.pack(pady=5, fill="x")

        censor_options = ["Gaussian", "Pixelized", "Fill", "Detect"]
        censor_privacy = ctk.CTkOptionMenu(
            master=self.sidebar,
            values=censor_options,
            command=self.on_censor_select,
        )

        censor_privacy.pack(padx=20, pady=5, fill="x")

        privacy_label = ctk.CTkLabel(self.sidebar, text="Privacy type")
        privacy_label.pack(pady=5, fill="x")

        privacy_options = ["All", "Specific", "Exclude"]

        select_privacy = ctk.CTkOptionMenu(
            master=self.sidebar,
            values=privacy_options,
            command=self.on_privacy_select,
        )

        select_privacy.pack(padx=20, pady=5, fill="x")

        self.record_btn = ctk.CTkButton(
            self.sidebar,
            text="Record",
            fg_color="#FF0000",
            hover_color="#990000",
            command=lambda: self.on_record(self.record_btn),
        )
        self.record_btn.pack_forget()

        self.record_label = ctk.CTkLabel(self.sidebar, text="Recording..")
        self.record_label.pack_forget()

        self.id_list_frame = ScrollableLabelButtonFrame(
            self.sidebar, command=self.on_remove_id
        )
        self.id_list_frame.pack(padx=20, pady=5, fill="x")

        self.extra_wrapper = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.extra_wrapper.pack_forget()

        check_btn = ctk.CTkButton(
            self.extra_wrapper, text="Output", command=self.on_output_check
        )
        check_btn.pack(padx=20, pady=5, fill="x")

        self.record_btn.bind("<Enter>", self.on_enter)
        self.record_btn.bind("<Leave>", self.on_leave)
        self.feed.bind("<Button-1>", self.on_feed_click)

        # Start the webcam thread
        webcam_thread = threading.Thread(target=self.webcam_thread)
        webcam_thread.start()

        self.update_feed()
        self.update_record_btn()

    def on_remove_id(self, item):
        item = int(item)
        if item in self.occultus.get_ids():
            self.id_list_frame.remove_item(item)
            self.occultus.remove_id(item)

    def on_privacy_select(self, value: str):
        self.occultus.set_privacy_control(value.lower())

    def on_censor_select(self, value: str):
        keys = {
            "Gaussian": "gaussian",
            "Pixelized": "pixel",
            "Fill": "fill",
            "Detect": "default",
        }

        if keys[value] == "default":
            self.occultus.set_blur_type(keys[value], show_label=True)
        else:
            self.occultus.set_blur_type(keys[value], show_label=False)

    def is_point_inside_box(self, point, box, padding=50):
        x, y = point
        x_min, y_min, x_max, y_max = box
        return (x_min - padding) <= x <= (x_max + padding) and (
            y_min - padding
        ) <= y <= (y_max + padding)

    def on_feed_click(self, event):
        # Get the coordinates of the mouse click relative to the label
        coords = (event.x, event.y)

        if self.dets is not None:
            for det in self.dets:
                if self.is_point_inside_box(coords, det["box"]):
                    if det["id"] not in self.occultus.get_ids():
                        self.occultus.add_id(int(det["id"]))
                        self.id_list_frame.add_item(det["id"])

    def update_feed(self):
        # Update the GUI every 5 milliseconds
        self.after(4, self.update_feed)

        # Update the Tkinter GUI with the latest frame
        if hasattr(self, "current_frame"):
            # if self.is_recording:
            #     self.occultus.save_video(frame=self.raw_frame, iterables=self.iterables)
            self.feed.configure(text="", image=self.current_frame)

    def update_record_btn(self):
        self.after(500, self.update_record_btn)

        if self.thread_started:
            self.record_btn.pack(padx=20, pady=5, fill="x")

    def on_record(self, parent: ctk.CTkButton):
        if self.is_recording:
            self.is_recording = False
            parent.configure(text="Start")
            self.extra_wrapper.pack(pady=5, fill="x")
            self.record_label.pack_forget()
        else:
            self.is_recording = True

            self.extra_wrapper.pack_forget()
            self.record_label.pack(fill="x")

            parent.configure(text="Stop")

    def on_output_check(self):
        subprocess.run(["explorer", "webcam_output"])

    def webcam_thread(self):
        # Open the video capture
        self.occultus = Occultus("weights/kamukha-v2.pt")
        self.occultus.set_blur_type("default", show_label=True)
        self.dets = []
        imgtk = None

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Failed to load webcam")

        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_dir = "webcam_output"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename("webcam.avi"))

        cap.release()

        writer = None

        for og_img, bboxes in self.occultus.detect_input_generator(frame_interval=2):
            if not self.running:
                break

            if not self.thread_started:
                self.thread_started = True

            self.dets = bboxes
            img = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            imgtk = ctk.CTkImage(img, size=(640, 480))
            self.current_frame = imgtk

            if self.is_recording:
                if isinstance(writer, cv2.VideoWriter):
                    writer.write(og_img)
                else:
                    writer = cv2.VideoWriter(
                        save_path,
                        cv2.VideoWriter_fourcc(*"XVID"),
                        int(
                            fps * 0.75
                        ),  # Match FPS with inference speed (75% of original fps)
                        (width, height),
                    )
                    writer.write(og_img)
            else:
                if isinstance(writer, cv2.VideoWriter):
                    writer.release()
                    writer = None

        if isinstance(writer, cv2.VideoWriter):
            writer.release()

    def on_close(self):
        # Release the video feed and close the window
        self.running = False
        self.controller.on_close()

    def on_enter(self, event):
        # Change cursor style on hover
        event.widget.configure(cursor="hand2")

    def on_leave(self, event):
        # Change cursor style back to the default
        event.widget.configure(cursor="")


class ScrollableLabelButtonFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, command=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self.command = command
        self.frame_list = []

    def add_item(self, item, image=None):
        if not isinstance(item, str):
            item = str(item)

        frame = ItemFrame(self, item, image, self.command)
        frame.pack(fill="x", pady=(0, 10), padx=5, anchor="w")
        self.frame_list.append(frame)

    def remove_item(self, item):
        for frame in self.frame_list:
            if not isinstance(item, str):
                item = str(item)

            if item == frame.label.cget("text"):
                frame.destroy()
                self.frame_list.remove(frame)
                return


class ItemFrame(ctk.CTkFrame):
    def __init__(self, master, item, image, command=None, **kwargs):
        super().__init__(master, **kwargs)

        self.label = ctk.CTkLabel(
            self, text=item, image=image, compound="left", padx=5, anchor="w"
        )
        self.button = ctk.CTkButton(self, text="Remove", width=100, height=24)
        if command is not None:
            self.button.configure(command=lambda: command(item))

        self.label.pack(side="left", fill="both")
        self.button.pack(side="right", padx=5)
