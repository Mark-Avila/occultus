import customtkinter as ctk

from interface.pages.select_stream import *
from interface.pages.landing import *


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
            center_frame,
            text="Load video",
            height=40,
            width=256,
            command=lambda: controller.show_frame("SelectFile"),
        )
        loadvid_button.pack(pady=20)

        loadstream_button = ctk.CTkButton(
            center_frame,
            text="Load stream",
            height=40,
            width=256,
            command=lambda: controller.show_frame("SelectStream"),
        )
        loadstream_button.pack()

        back_button = ctk.CTkButton(
            self,
            text="Back",
            fg_color="transparent",
            command=lambda: controller.show_frame("Landing"),
        )
        back_button.pack(pady=20)

        loadvid_button.bind("<Enter>", self.on_enter)
        loadvid_button.bind("<Leave>", self.on_leave)
        loadstream_button.bind("<Enter>", self.on_enter)
        loadstream_button.bind("<Leave>", self.on_leave)
        back_button.bind("<Enter>", self.on_enter)
        back_button.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        # Change cursor style on hover
        event.widget.configure(cursor="hand2")

    def on_leave(self, event):
        # Change cursor style back to the default
        event.widget.configure(cursor="")

    # def open_videopage(self, controller):
    #     # Create an instance of SelectStreamPageWindow
    #     video_window = VideoPage(controller=controller)

    #     # Make the new window modal
    #     video_window.grab_set()

    #     # Wait for the new window to be closed before continuing
    #     self.wait_window(video_window)
