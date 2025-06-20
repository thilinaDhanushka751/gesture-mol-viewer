import os
import tkinter as tk
from tkinter import messagebox
from pymol_renderer import PyMolRenderer
from gestures.hand_control import GestureController  # Still valid for gesture control

MOLECULE_FOLDER = "molecules"

class MoleculeApp:
    def __init__(self, master):
        self.master = master
        self.master.title("🧪 Molecular Viewer")

        # PyMOL controller
        self.pymol = PyMolRenderer()

        # Sidebar for molecule list
        self.sidebar = tk.Frame(master, width=200)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        tk.Label(self.sidebar, text="Available Molecules:").pack(anchor="w")

        self.molecule_listbox = tk.Listbox(self.sidebar, height=20, width=30)
        self.molecule_listbox.pack(pady=5)
        self.load_molecule_names()

        self.load_button = tk.Button(self.sidebar, text="🔬 Visualize Molecule", command=self.load_selected_molecule)
        self.load_button.pack(pady=10)

        self.gesture_button = tk.Button(self.sidebar, text=" Start Gesture Control", command=self.start_gesture_control)
        self.gesture_button.pack(pady=5)

        # Controls panel
        self.controls = tk.Frame(master)
        self.controls.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        tk.Button(self.controls, text="🔄 Rotate X", command=lambda: self.pymol.rotate("x")).pack(pady=5)
        tk.Button(self.controls, text="🔄 Rotate Y", command=lambda: self.pymol.rotate("y")).pack(pady=5)
        tk.Button(self.controls, text="🔄 Rotate Z", command=lambda: self.pymol.rotate("z")).pack(pady=5)
        tk.Button(self.controls, text="➕ Zoom In", command=lambda: self.pymol.zoom(1.2)).pack(pady=5)
        tk.Button(self.controls, text="➖ Zoom Out", command=lambda: self.pymol.zoom(0.8)).pack(pady=5)

        master.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_molecule_names(self):
        files = os.listdir(MOLECULE_FOLDER)
        mol_files = [f for f in files if f.endswith(".mol") or f.endswith(".pdb")]
        mol_files.sort()  # Optional: sort alphabetically
        self.molecule_listbox.delete(0, tk.END)
        for file in mol_files:
            self.molecule_listbox.insert(tk.END, file)

    def load_selected_molecule(self):
        selection = self.molecule_listbox.curselection()
        if not selection:
            messagebox.showwarning("No selection", "Please select a molecule from the list.")
            return

        file_name = self.molecule_listbox.get(selection[0])
        file_path = os.path.join(MOLECULE_FOLDER, file_name)
        self.pymol.load_molecule(file_path)

    def start_gesture_control(self):
        controller = GestureController()
        controller.detect(
            on_swipe_left=lambda: self.pymol.rotate("y", -15),
            on_swipe_right=lambda: self.pymol.rotate("y", 15),
            on_zoom_in=lambda: self.pymol.zoom(1.2),
            on_zoom_out=lambda: self.pymol.zoom(0.8),
        )

    def on_close(self):
        self.pymol.quit()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MoleculeApp(root)
    root.mainloop()
