import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from occultus.core import Occultus

cap = None


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("720x480")

        self.detect = Occultus("weights/kamukha-v2.pt")
        self.detect.load_stream()

        self.frames = self.detect.initialize()

        self.current_image = None

        self.content = ctk.CTkLabel(self, fg_color="#000000", text="")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.rowconfigure(0, weight=1)

        button = ctk.CTkButton(self.sidebar, text="Test", command=self.button_callbck)
        button.pack(padx=20, pady=20)

        self.video_loop_temp()

    def button_callbck(self):
        print("button clicked")

    def video_loop(self):
        """Get frame from the video stream and show it in Tkinter"""
        ok, frame = self.vs.read()  # read frame from video stream

        frame = cv2.flip(frame, 1)

        if ok:  # frame captured without any errors
            cv2image = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGBA
            )  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(
                image=self.current_image
            )  # convert image for tkinter
            self.content.imgtk = (
                imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            )
            self.content.configure(image=imgtk)  # show the image
        self.after(30, self.video_loop)

    def video_loop_temp(self):
        for pred, dataset, iterables in self.detect.inference(self.frames):
            frame = self.detect.process(pred, dataset, iterables)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = Image.fromarray(frame)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=frame)  # convert image for tkinter
            self.content.imgtk = (
                imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            )
            self.content.configure(image=imgtk)  # show the image
            self.after(10, self.video_loop_temp)

            return


app = App()
app.mainloop()
