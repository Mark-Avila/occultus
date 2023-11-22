import customtkinter as ctk


class Sidebar(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.create_widgets()

    def create_widgets(self):
        button1 = ctk.CTkButton(self, text="Option 1", command=self.button_callbck)
        button2 = ctk.CTkButton(self, text="Option 2", command=self.button_callbck)
        button3 = ctk.CTkButton(self, text="Option 3", command=self.button_callbck)

        # Arrange buttons vertically
        button1.pack(fill=ctk.X)
        button2.pack(fill=ctk.X)
        button3.pack(fill=ctk.X)

    def button_callbck(self):
        print("button clicked")


class App(ctk.CTkFrame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Example content in the main area
        label = ctk.CTkLabel(self, text="Main Content Area")
        label.pack()

        # Create a sidebar instance
        sidebar = Sidebar(self, width=200)
        sidebar.pack(side=ctk.LEFT, fill=ctk.Y)


root = ctk.CTk()
app = App(master=root)
app.mainloop()
