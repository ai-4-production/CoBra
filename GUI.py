import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os

class ProductionLayoutGUI:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Production Layout Simulation")
        self.create_widgets()

    def create_widgets(self):
        self.title_label = tk.Label(self.window, text="Production Layout Simulation")
        self.title_label.pack()

        self.radio_frame = tk.Frame(self.window)
        self.radio_frame.pack()

        self.radio_choice = tk.StringVar()

        self.use_existing_layout = tk.Radiobutton(self.radio_frame, text="Use existing production layout", variable=self.radio_choice, value="use_existing", command=self.show_existing_layouts)
        self.use_existing_layout.pack()

        self.new_layout = tk.Radiobutton(self.radio_frame, text="Create a new production layout", variable=self.radio_choice, value="create_new", command=self.show_new_layout)
        self.new_layout.pack()

        self.existing_layouts_frame = tk.Frame(self.window)
        self.layout_choice_label = tk.Label(self.existing_layouts_frame, text="Choose a production layout:")
        self.layout_choice_label.pack(side=tk.LEFT)

        self.layout_choice_var = tk.StringVar(self.existing_layouts_frame)
        self.layout_choice_var.set("")

        self.layout_choice_dropdown = tk.OptionMenu(self.existing_layouts_frame, self.layout_choice_var, "")
        self.layout_choice_dropdown.pack(side=tk.LEFT)

        self.load_layout_button = tk.Button(self.existing_layouts_frame, text="Load Layout", command=self.load_existing_layout)
        self.load_layout_button.pack(side=tk.LEFT)

        self.existing_layouts_frame.pack()

        self.new_layout_frame = tk.Frame(self.window)

        self.cell_type_label = tk.Label(self.new_layout_frame, text="Cell Type:")
        self.cell_type_label.grid(row=0, column=0)

        self.cell_type_var = tk.StringVar(self.new_layout_frame)
        self.cell_type_var.set("Man")

        self.cell_type_dropdown = tk.OptionMenu(self.new_layout_frame, self.cell_type_var, "Man", "Dis")
        self.cell_type_dropdown.grid(row=0, column=1)

        self.machines_label = tk.Label(self.new_layout_frame, text="Machines:")
        self.machines_label.grid(row=1, column=0)

        self.machines_entry = tk.Entry(self.new_layout_frame)
        self.machines_entry.grid(row=1, column=1)

        self.agents_label = tk.Label(self.new_layout_frame, text="Agents:")
        self.agents_label.grid(row=2, column=0)

        self.agents_entry = tk.Entry(self.new_layout_frame)
        self.agents_entry.grid(row=2, column=1)

        self.storage_cap_label = tk.Label(self.new_layout_frame, text="Storage Capacity:")
        self.storage_cap_label.grid(row=3, column=0)

        self.storage_cap_entry = tk.Entry(self.new_layout_frame)
        self.storage_cap_entry.grid(row=3, column=1)

        self.input_cap_label = tk.Label(self.new_layout_frame, text="Input Capacity:")
        self.input_cap_label.grid(row=4, column=0)

        self.input_cap_entry = tk.Entry(self.new_layout_frame)
        self.input_cap_entry.grid(row=4, column=1)

        self.output_cap_label = tk.Label(self.new_layout_frame, text="Output Capacity:")
        self.output_cap_label.grid(row=5, column=0)

        self.output_cap_entry = tk.Entry(self.new_layout_frame)
        self.output_cap_entry.grid(row=5, column=1)

        self.parent_cell_label = tk.Entry(self.new_layout_frame, text="Parent Cell:")
        self.parent_cell_label.grid(row=6, column=0)

        self.parent_cell_entry = tk.Entry(self.new_layout_frame)
        self.parent_cell_entry.grid(row=6, column=1)

        self.cell_level_label = tk.Label(self.new_layout_frame, text="Cell Level:")
        self.cell_level_label.grid(row=7, column=0)

        self.cell_level_entry = tk.Entry(self.new_layout_frame)
        self.cell_level_entry.grid(row=7, column=1)

        self.add_cell_button = tk.Button(self.new_layout_frame, text="Add Cell", command=self.add_new_cell)
        self.add_cell_button.grid(row=8, column=0)

        self.remove_cell_button = tk.Button(self.new_layout_frame, text="Remove Cell", command=self.remove_cell)
        self.remove_cell_button.grid(row=8, column=1)

        self.new_layout_frame.pack()

        self.window.mainloop()

    def show_existing_layouts(self):
        self.new_layout_frame.pack_forget()
        self.existing_layouts_frame.pack()
        self.load_existing_layouts()

    def show_new_layout(self):
        self.existing_layouts_frame.pack_forget()
        self.new_layout_frame.pack()

    def load_existing_layouts(self):
        self.layout_choice_dropdown['menu'].delete(0, 'end')
        setup_folder_path = os.path.join(os.getcwd(), "setups")
        print(setup_folder_path)
        for filename in os.listdir(setup_folder_path):
            if filename.endswith(".csv"):
                layout = pd.read_csv(os.path.join(setup_folder_path, filename))
                self.layout_choice_dropdown['menu'].add_command(label=filename, command=tk._setit(self.layout_choice_var, filename))
                self.layout_choice_var.set(layout.columns[0])

    def load_existing_layout(self):
        file_path = filedialog.askopenfilename(initialdir="./layouts", title="Select a layout file", filetypes=[("TXT Files", "*.txt")])
        if file_path:
            layout = pd.read_csv(file_path)
            # Code to load the layout into the simulation goes here

    def add_new_cell(self):
        cell_type = self.cell_type_var.get()
        machines = self.machines_entry.get()
        agents = self.agents_entry.get()
        storage_cap = self.storage_cap_entry.get()
        input_cap = self.input_cap_entry.get()
        output_cap = self.output_cap_entry.get()
        parent_cell = self.parent_cell_entry.get()
        cell_level = self.cell_level_entry.get()

        # Code to add the new cell to the layout goes here

    def remove_cell(self):
        # Code to remove a cell from the layout goes here
        pass

if __name__ == "__main__":    
    ProductionLayoutGUI()