import customtkinter as ctk
import cv2
from PIL import Image, ImageTk


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        self.geometry("1080x720")

        container = ctk.CTkFrame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.pages = {}

        for Page in (LandingPage, SelectInputPage, SelectStreamPage, StreamPage):
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
            command=lambda: controller.show_frame(StreamPage),
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


class StreamPage(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTk, controller):
        ctk.CTkFrame.__init__(self, parent)

        self.vid = None

        CONTENT_PADDING = 140
        RECORD_BTN_SIZE = 60

        controller.protocol("WM_DELETE_WINDOW", lambda: self.on_close(controller))

        # self.content = ctk.CTkLabel(
        #     self, fg_color="#000000", text="", width=640, height=480
        # )
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=0, column=1, sticky="nsew")

        self.columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.rowconfigure(0, weight=1)

        self.feed = ctk.CTkLabel(
            self.content, fg_color="#000000", text="", width=640, height=480
        )
        self.feed.pack(padx=CONTENT_PADDING, pady=(40, 0))

        wrapper = ctk.CTkFrame(self.content, fg_color="transparent")
        wrapper.pack(fill="x", padx=CONTENT_PADDING, pady=(40, 0))

        record_btn = ctk.CTkButton(
            wrapper,
            corner_radius=50,
            width=RECORD_BTN_SIZE,
            height=RECORD_BTN_SIZE,
            text=">",
            command=self.start,
        )
        record_btn.pack(side=ctk.LEFT)

        time_lbl = ctk.CTkLabel(
            wrapper, text="0:00", font=ctk.CTkFont(weight="bold", size=24)
        )
        time_lbl.pack(side=ctk.LEFT, padx=(40, 0))

        censor_label = ctk.CTkLabel(self.sidebar, text="Censor type")
        censor_label.pack(pady=5)

        censor_options = ["Gaussian", "Pixel", "Fill", "Default"]

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

        # detect_btn = ctk.CTkButton(self.sidebar, text="Detect", command=self.start)
        # detect_btn.pack(padx=20, pady=5)

        self.feed.bind("<Button-1>", self.on_feed_click)
        self.after(1, self.start)

    def on_privacy_select(self, value: str):
        # self.detect.set_privacy_control(value)
        print(value)

    def on_censor_select(self, value: str):
        # self.detect.set_privacy_control(value)
        print(value)

    def on_feed_click(self, event):
        # Get the coordinates of the mouse click relative to the label
        x, y = event.x, event.y
        print(f"Mouse clicked at ({x}, {y}) within the label.")

    def start(self):
        if self.vid is None:
            self.vid = cv2.VideoCapture(0)
            self.update()

    def update(self):
        # Get the latest frame from the video feed
        ret, frame = self.vid.read()

        if ret:
            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to PIL Image
            frame = Image.fromarray(frame)
            imgtk = ctk.CTkImage(frame, size=(640, 480))  # convert image for tkinter
            self.feed.imgtk = (
                imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            )
            self.feed.configure(image=imgtk)  # show the image
        self.after(10, self.update)

    def on_close(self, parent: ctk.CTk):
        # Release the video feed and close the window
        if self.vid and self.vid.isOpened():
            self.vid.release()
        parent.destroy()


if __name__ == "__main__":
    app = App()
    app.resizable(False, False)
    app.mainloop()
