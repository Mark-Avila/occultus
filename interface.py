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
        self.detect = Occultus("weights/kamukha-v2.pt")

        # Set model for streaming (using camera)
        self.detect.load_stream()

        # Set model configurations
        self.detect.set_config({"blur_type": "pixel", "flipped": True})

        # Initialize library (eg. ready library for GPU inferencing)
        self.frames = self.detect.initialize()

        # Initialize current_image for storing frames
        self.current_image = None

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

        # Start detection
        self.video_loop()

    def set_blur(self):
        self.detect.set_config({"blur_type": "blur"})

    def set_pixel(self):
        self.detect.set_config({"blur_type": "pixel"})

    def set_fill(self):
        self.detect.set_config({"blur_type": "fill"})

    def set_detect(self):
        self.detect.set_config({"blur_type": "detect"})

    def video_loop(self):
        for pred, dataset, iterables in self.detect.inference(self.frames):
            frame = self.detect.process(pred, dataset, iterables)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = Image.fromarray(frame)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=frame)  # convert image for tkinter
            self.content.imgtk = (
                imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            )
            self.content.configure(image=imgtk)  # show the image
            self.after(10, self.video_loop)

            return


app = App()
app.mainloop()
