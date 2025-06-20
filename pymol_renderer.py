import pymol
from pymol import cmd
import threading
from rdkit import Chem
from pymol import cmd

class PyMolRenderer:

    def highlight_chiral_centers(file_path):
        mol = Chem.MolFromMolFile(file_path)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)

        if not chiral_centers:
            print("No chiral centers found.")
            return

        atom_ids = [str(idx) for idx, _ in chiral_centers]
        selection = "+".join(atom_ids)
        cmd.select("chiral_centers", f"ID {selection}")
        cmd.show("spheres", "chiral_centers")
        cmd.color("orange", "chiral_centers")
        print("✅ Highlighted chiral centers:", atom_ids)

    def __init__(self):
        self.thread = threading.Thread(target=self._launch_pymol)
        self.thread.start()

    def _launch_pymol(self):
        pymol.finish_launching(['pymol', '-A1'])
                              


    def load_molecule(self, file_path):
        cmd.reinitialize()
        cmd.load(file_path, "mol")
        cmd.show("sticks", "mol")
        cmd.zoom("mol")

    def rotate(self, axis, angle=15):
        if "mol" in cmd.get_names("all"):
            cmd.rotate(axis, angle, "mol")

    def zoom(self, factor=1.2):
        if "mol" in cmd.get_names("all"):
            cmd.zoom("mol", factor)

    def quit(self):
        cmd.quit()


from rdkit import Chem
from pymol import cmd

def highlight_chiral_centers(file_path):
    mol = Chem.MolFromMolFile(file_path)
    if not mol:
        print("Could not parse molecule.")
        return

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)

    if not chiral_centers:
        print("No chiral centers found.")
        return

    for atom_idx, chirality in chiral_centers:
        selection_name = f"chiral_{atom_idx}"
        cmd.select(selection_name, f"ID {atom_idx}")
        cmd.show("sticks", selection_name)
        cmd.set("stick_radius", 0.15, selection=selection_name)
        cmd.color("orange", selection_name)
        cmd.label(selection_name, f"'{chirality}'")

    ids = [f"{idx} ({ch})" for idx, ch in chiral_centers]
    print("✅ Highlighted chiral centers:", ids)
