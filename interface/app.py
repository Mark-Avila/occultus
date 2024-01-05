import customtkinter as ctk

from interface.pages.landing import *
from interface.pages.select_input import *
from interface.pages.select_stream import *
from interface.pages.select_file import *


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        self.geometry("720x480")
        self.title("Occultus UI")

        container = ctk.CTkFrame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.pages = {}

        for Page, page_name in zip(
            (LandingPage, SelectInputPage, SelectStreamPage, SelectFilePage),
            ("Landing", "SelectInput", "SelectStream", "SelectFile"),
        ):
            frame = Page(container, self)
            self.pages[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("Landing")

    def show_frame(self, cont):
        frame = self.pages[cont]
        frame.tkraise()

    def on_close(self):
        self.destroy()
