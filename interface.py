import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from occultus.core import Occultus

cap = None


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("720x480")

        # Initialize Occultus object and assign to detect
        self.detect = Occultus("weights/kamukha-v3.pt")

        # Set model for streaming (using camera)
        self.detect.load_stream()

        # Set model configurations
        self.detect.set_config(
            {"blur_type": "pixel", "flipped": True, "conf_thres": 0.5}
        )

        self.detect.set_blur_type("default")

        # Initialize library (eg. ready library for GPU inferencing)
        self.frames = self.detect.initialize()

        # Initialize current_image for storing frames
        self.bboxes = None

        # Interace related code
        self.content = ctk.CTkLabel(self, fg_color="#000000", text="")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.rowconfigure(0, weight=1)

        blur_button = ctk.CTkButton(self.sidebar, text="Blur", command=self.set_blur)
        blur_button.pack(padx=20, pady=5)

        pixel_button = ctk.CTkButton(self.sidebar, text="Pixel", command=self.set_pixel)
        pixel_button.pack(padx=20, pady=5)

        fill_button = ctk.CTkButton(self.sidebar, text="Fill", command=self.set_fill)
        fill_button.pack(padx=20, pady=5)

        detect_button = ctk.CTkButton(
            self.sidebar, text="Detect", command=self.set_detect
        )
        detect_button.pack(padx=20, pady=5)

        privacy_options = ["all", "specific", "exclude"]

        select_privacy = ctk.CTkOptionMenu(
            master=self.sidebar,
            values=privacy_options,
            command=self.on_privacy_select,
        )

        select_privacy.pack(padx=20, pady=5)

        self.content.bind("<Button-1>", self.on_camera_click)

        # Start detection
        self.video_loop()

    def on_privacy_select(self, value: str):
        self.detect.set_privacy_control(value)

    def set_blur(self):
        self.detect.set_blur_type("gaussian")

    def set_pixel(self):
        self.detect.set_blur_type("pixel")

    def set_fill(self):
        self.detect.set_blur_type("fill")

    def set_detect(self):
        self.detect.set_blur_type("default")

    def is_point_inside_box(self, point, box, padding=50):
        x, y = point
        x_min, y_min, x_max, y_max = box
        return (x_min - padding) <= x <= (x_max + padding) and (
            y_min - padding
        ) <= y <= (y_max + padding)

    def on_camera_click(self, event):
        # Get the coordinates of the mouse click relative to the label
        coords = (event.x, event.y)

        for det in self.dets:
            print(f"Checking bounding box: {det['box']}")
            if self.is_point_inside_box(coords, det["box"]):
                print(f"Click coordinates {coords} are inside object ID {det['id']}")
                self.detect.append_id(det["id"])
                continue
            else:
                print(
                    f"Click coordinates {coords} are outside bounding box {det['box']}"
                )

    def video_loop(self):
        for pred, dataset, iterables in self.detect.inference(self.frames):
            [frame, dets] = self.detect.process(pred, dataset, iterables)

            self.dets = dets

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = Image.fromarray(frame)  # convert image for PIL
            imgtk = ctk.CTkImage(frame, size=(640, 480))  # convert image for tkinter
            self.content.imgtk = (
                imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            )
            self.content.configure(image=imgtk)  # show the image
            self.after(5, self.video_loop)

            return


app = App()
app.resizable(False, False)
app.mainloop()
