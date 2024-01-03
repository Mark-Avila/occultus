import customtkinter as ctk


class SelectFilePage(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTk, controller):
        ctk.CTkFrame.__init__(self, parent)

        # Create a frame to center the content
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.pack(expand=True)

        select_label = ctk.CTkLabel(
            center_frame, text="Select file", font=ctk.CTkFont("Helvetica", 16)
        )
        select_label.pack()

        file_input = ctk.CTkEntry(center_frame, width=256, height=32)
        file_input.pack(pady=(20, 5))

        file_label = ctk.CTkLabel(center_frame, text="File path")
        file_label.pack()

        file_input_button = ctk.CTkButton(
            center_frame, text="Start", width=256, height=32
        )
        file_input_button.pack(pady=(5, 0))

        or_label = ctk.CTkLabel(center_frame, text="OR")
        or_label.pack(pady=30)

        select_file_button = ctk.CTkButton(
            center_frame,
            text="Select File",
            command=lambda: self.open_streampage(controller=controller),
            width=256,
            height=32,
        )
        select_file_button.pack(pady=(0, 5))
        select_label = ctk.CTkLabel(center_frame, text="Select file with File Browser")
        select_label.pack()

        back_button = ctk.CTkButton(
            self,
            text="Back",
            fg_color="transparent",
            command=lambda: controller.show_frame("SelectInput"),
        )
        back_button.pack(pady=20)

        file_input_button.bind("<Enter>", self.on_enter)
        file_input_button.bind("<Leave>", self.on_leave)
        select_file_button.bind("<Enter>", self.on_enter)
        select_file_button.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        # Change cursor style on hover
        event.widget.configure(cursor="hand2")

    def on_leave(self, event):
        # Change cursor style back to the default
        event.widget.configure(cursor="")
