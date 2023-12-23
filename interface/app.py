import customtkinter as ctk

from interface.pages.landing import *
from interface.pages.select_input import *
from interface.pages.select_stream import *


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
