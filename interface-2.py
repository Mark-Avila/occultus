import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from occultus.core import Occultus
import threading


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        self.geometry("720x480")

        container = ctk.CTkFrame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.pages = {}

        for Page in (LandingPage, SelectInputPage, SelectStreamPage):
            frame = Page(container, self)
            self.pages[Page] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(LandingPage)

    def show_frame(self, cont):
        frame = self.pages[cont]
        frame.tkraise()


class LandingPage(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTk, controller):
        ctk.CTkFrame.__init__(self, parent)

        # Create a frame to center the content
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.pack(expand=True)

        # Add your content to the center frame
        label1 = ctk.CTkLabel(
            center_frame,
            text="Occultus",
            font=ctk.CTkFont(family="Helvetica", size=48),
            fg_color=None,
        )
        label1.pack()

        start_button = ctk.CTkButton(
            center_frame,
            text="Start",
            font=ctk.CTkFont(family="Helvetica", size=18),
            width=256,
            height=48,
            command=lambda: controller.show_frame(SelectInputPage),
        )
        start_button.bind("<Enter>", self.on_enter)
        start_button.bind("<Leave>", self.on_leave)
        start_button.pack(pady=28)

    def on_enter(self, event):
        # Change cursor style on hover
        event.widget.configure(cursor="hand2")

    def on_leave(self, event):
        # Change cursor style back to the default
        event.widget.configure(cursor="")


class SelectInputPage(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTk, controller):
        ctk.CTkFrame.__init__(self, parent)

        # Create a frame to center the content
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.pack(expand=True)

        # Add your content to the center frame
        select_label = ctk.CTkLabel(
            center_frame,
            text="Select input",
            font=ctk.CTkFont(family="Helvetica", size=16),
            fg_color=None,
        )
        select_label.pack()

        loadvid_button = ctk.CTkButton(
            center_frame, text="Load video", height=40, width=256
        )
        loadvid_button.pack(pady=20)

        loadstream_button = ctk.CTkButton(
            center_frame,
            text="Load stream",
            height=40,
            width=256,
            command=lambda: controller.show_frame(SelectStreamPage),
        )
        loadstream_button.pack()

        loadvid_button.bind("<Enter>", self.on_enter)
        loadvid_button.bind("<Leave>", self.on_leave)
        loadstream_button.bind("<Enter>", self.on_enter)
        loadstream_button.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        # Change cursor style on hover
        event.widget.configure(cursor="hand2")

    def on_leave(self, event):
        # Change cursor style back to the default
        event.widget.configure(cursor="")


class SelectStreamPage(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTk, controller):
        ctk.CTkFrame.__init__(self, parent)

        # Create a frame to center the content
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.pack(expand=True)

        select_label = ctk.CTkLabel(
            center_frame, text="Input stream type", font=ctk.CTkFont("Helvetica", 16)
        )
        select_label.pack()

        stream_input = ctk.CTkEntry(center_frame, width=256, height=32)
        stream_input.pack(pady=(20, 5))
        stream_label = ctk.CTkLabel(center_frame, text="Device link")
        stream_label.pack()
        stream_start_button = ctk.CTkButton(
            center_frame, text="Start", width=256, height=32
        )
        stream_start_button.pack(pady=(5, 0))

        or_label = ctk.CTkLabel(center_frame, text="OR")
        or_label.pack(pady=30)

        camera_button = ctk.CTkButton(
            center_frame,
            text="Camera",
            command=self.open_stream_window,
            width=256,
            height=32,
        )
        camera_button.pack(pady=(0, 5))
        camera_label = ctk.CTkLabel(center_frame, text="Use device camera instead")
        camera_label.pack()

        stream_start_button.bind("<Enter>", self.on_enter)
        stream_start_button.bind("<Leave>", self.on_leave)
        camera_button.bind("<Enter>", self.on_enter)
        camera_button.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        # Change cursor style on hover
        event.widget.configure(cursor="hand2")

    def on_leave(self, event):
        # Change cursor style back to the default
        event.widget.configure(cursor="")

    def open_stream_window(self):
        # Create an instance of SelectStreamPageWindow
        stream_window = StreamPage()
        stream_window.title("Select Stream Window")

        # Make the new window modal
        stream_window.grab_set()

        # Wait for the new window to be closed before continuing
        self.wait_window(stream_window)


class StreamPage(ctk.CTkToplevel):
    def __init__(self):
        ctk.CTkToplevel.__init__(self)
        self.title("Occultus")

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

        # Video feed display
        self.feed = ctk.CTkLabel(
            self.content, fg_color="#000000", text="Loading", width=640, height=480
        )
        self.feed.pack()

        censor_label = ctk.CTkLabel(self.sidebar, text="Censor type")
        censor_label.pack(pady=5)

        censor_options = ["Gaussian", "Pixelized", "Fill", "Detect"]
        censor_privacy = ctk.CTkOptionMenu(
            master=self.sidebar,
            values=censor_options,
            command=self.on_censor_select,
        )

        censor_privacy.pack(padx=20, pady=5)

        privacy_label = ctk.CTkLabel(self.sidebar, text="Privacy type")
        privacy_label.pack(pady=5)

        privacy_options = ["All", "Specific", "Exclude"]

        select_privacy = ctk.CTkOptionMenu(
            master=self.sidebar,
            values=privacy_options,
            command=self.on_privacy_select,
        )

        select_privacy.pack(padx=20, pady=5)

        record_btn = ctk.CTkButton(
            self.sidebar,
            text="Record",
            fg_color="#FF0000",
            hover_color="#990000",
            command=lambda: self.on_record(record_btn),
        )
        record_btn.pack(padx=20, pady=5)

        record_btn.bind("<Enter>", self.on_enter)
        record_btn.bind("<Leave>", self.on_leave)
        self.feed.bind("<Button-1>", self.on_feed_click)

        # Start the webcam thread
        webcam_thread = threading.Thread(target=self.webcam_thread)
        webcam_thread.start()

        self.update_feed()

    def on_privacy_select(self, value: str):
        self.occultus.set_privacy_control(value.lower())

    def on_censor_select(self, value: str):
        keys = {
            "None": "default",
            "Pixelized": "pixel",
            "Fill": "fill",
            "Detect": "default",
        }

        self.occultus.set_blur_type(keys[value])

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
                print(f"Checking bounding box: {det['box']}")
                if self.is_point_inside_box(coords, det["box"]):
                    print(f"Click is inside object ID: {det['id']}")
                    self.occultus.append_id(det["id"])
                    continue

    def update_feed(self):
        # Update the GUI every 5 milliseconds
        self.after(4, self.update_feed)

        # Update the Tkinter GUI with the latest frame
        if hasattr(self, "current_frame"):
            # if self.is_recording:
            #     self.occultus.save_video(frame=self.raw_frame, iterables=self.iterables)
            self.feed.configure(text="", image=self.current_frame)

    def on_record(self, parent: ctk.CTkButton):
        if self.is_recording:
            self.is_recording = False
            parent.configure(text="Start")
        else:
            self.is_recording = True
            parent.configure(text="Stop")

    def webcam_thread(self):
        # Open the video capture
        self.occultus = Occultus("weights/kamukha-v3.pt")
        self.occultus.set_blur_type("pixel")
        self.dets = []
        imgtk = None

        for og_img, bboxes in self.occultus.detect_input_generator():
            if not self.running:
                break

            self.dets = bboxes
            img = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            imgtk = ctk.CTkImage(img, size=(640, 480))
            self.current_frame = imgtk

    def on_close(self):
        # Release the video feed and close the window
        self.running = False
        self.destroy()

    def on_enter(self, event):
        # Change cursor style on hover
        event.widget.configure(cursor="hand2")

    def on_leave(self, event):
        # Change cursor style back to the default
        event.widget.configure(cursor="")


if __name__ == "__main__":
    app = App()
    app.resizable(False, False)
    app.mainloop()
