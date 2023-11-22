import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

cap = None


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("720x480")

        # self.button = ctk.CTkButton(self, text="my button", command=self.button_callbck)
        # self.button.pack(padx=20, pady=20)
        self.vs = cv2.VideoCapture(
            0
        )  # capture video frames, 0 is your default video camera
        self.current_image = None

        self.content = ctk.CTkLabel(self, fg_color="#000000", text="")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.rowconfigure(0, weight=1)

        self.video_loop()

    def button_callbck(self):
        print("button clicked")

    def video_loop(self):
        """Get frame from the video stream and show it in Tkinter"""
        ok, frame = self.vs.read()  # read frame from video stream
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


app = App()
app.mainloop()
