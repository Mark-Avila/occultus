import customtkinter as ctk


class DetectPage(ctk.CTkFrame):
    def __init__(
        self, parent: ctk.CTk, controller, text="Detecting faces. Please wait"
    ):
        ctk.CTkFrame.__init__(self, parent)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(2, weight=1)

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.grid(row=1, column=1)

        message_label = ctk.CTkLabel(wrapper, text=text, font=ctk.CTkFont(size=24))
        message_label.pack()

        self.progress = ctk.CTkProgressBar(wrapper, height=12, width=512)
        self.progress.set(0)
        self.progress.pack(pady=20)

    def set_progress(self, new_value):
        self.progress.set(new_value)
