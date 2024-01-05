import customtkinter as ctk
from interface.windows.video import *
from tkinter import filedialog


class SelectFilePage(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTk, controller):
        ctk.CTkFrame.__init__(self, parent)

        self.source = ""

        # Create a frame to center the content
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.pack(expand=True)

        select_label = ctk.CTkLabel(
            center_frame, text="Input file type", font=ctk.CTkFont("Helvetica", 16)
        )
        select_label.pack()

        self.stream_input = ctk.CTkEntry(center_frame, width=256, height=32)
        self.stream_input.pack(pady=(20, 5))

        self.stream_label = ctk.CTkLabel(center_frame, text="File URL")
        self.stream_label.pack()

        stream_start_button = ctk.CTkButton(
            center_frame,
            text="Start",
            height=32,
            command=lambda: self.on_stream_start(controller=controller),
        )
        stream_start_button.pack(pady=(5, 0))

        or_label = ctk.CTkLabel(center_frame, text="OR")
        or_label.pack(pady=30)

        select_file_button = ctk.CTkButton(
            center_frame,
            text="Select File",
            width=256,
            height=32,
            command=self.on_select,
        )
        select_file_button.pack(pady=(0, 5))

        self.file_label = ctk.CTkLabel(center_frame, text="")
        self.file_label.pack()

        start_button = ctk.CTkButton(
            center_frame,
            text="Start",
            height=32,
            command=lambda: self.open_videopage(controller=controller),
        )
        start_button.pack(pady=(0, 5))

        back_button = ctk.CTkButton(
            self,
            text="Back",
            fg_color="transparent",
            command=lambda: controller.show_frame("SelectInput"),
        )
        back_button.pack(pady=20)

        # file_input_button.bind("<Enter>", self.on_enter)
        # file_input_button.bind("<Leave>", self.on_leave)
        select_file_button.bind("<Enter>", self.on_enter)
        select_file_button.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        # Change cursor style on hover
        event.widget.configure(cursor="hand2")

    def on_leave(self, event):
        # Change cursor style back to the default
        event.widget.configure(cursor="")

    def on_select(self):
        self.source = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mkv;*.mov;*.mpg;*.mpeg;*.m4v;*.wmv")
            ],
        )

        self.file_label.configure(
            text=f"Selected file: {os.path.basename(self.source)}", text_color="white"
        )

    def on_stream_start(self, controller):
        url_input = self.stream_input.get()

        vid_formats = [
            ".mov",
            ".avi",
            ".mp4",
            ".mpg",
            ".mpeg",
            ".m4v",
            ".wmv",
            ".mkv",
        ]  # acceptable video suffixes

        if any(url_input.endswith(ext) for ext in vid_formats):
            video_window = VideoPage(controller=controller, source=url_input)
            video_window.grab_set()
            self.wait_window(video_window)
        else:
            self.stream_label.configure(text="Invalid Video URL", text_color="red")

    def open_videopage(self, controller):
        if self.source:
            # Create an instance of SelectStreamPageWindow
            video_window = VideoPage(controller=controller, source=self.source)

            # Make the new window modal
            video_window.grab_set()

            # Wait for the new window to be closed before continuing
            self.wait_window(video_window)
        else:
            self.file_label.configure(text="Please select a file", text_color="red")
