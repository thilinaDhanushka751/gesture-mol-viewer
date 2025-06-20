# molecule_analyzer.py
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski

# Define common functional groups with SMARTS patterns
FUNCTIONAL_GROUPS = {
    "Hydroxyl (OH)": "[CX4][OH]",
    "Carboxylic Acid": "C(=O)[OH]",
    "Ketone (C=O)": "[CX3](=O)[#6]",
    "Amino (NH2)": "[NX3;H2,H1;!$(NC=O)]",
    "Nitro Group": "[NX3](=O)=O",
    "Alkene (C=C)": "C=C",
    "Alkyne (C#C)": "C#C",
}

# Define toxic groups with SMARTS patterns
TOXIC_GROUPS = {
    "Nitro": "[NX3](=O)=O",
    "Cyanide": "[C-]#N",
    "Epoxide": "C1OC1",
    "Azide": "N=[N+]=[N-]"
}

def analyze_molecule(file_path):
    try:
        if file_path.endswith(".mol"):
            mol = Chem.MolFromMolFile(file_path)
        elif file_path.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(file_path)
        else:
            return {"Error": "Unsupported file format."}

        if mol is None:
            return {"Error": "Could not parse molecule."}

        # Basic chemical properties
        info = {
            "Formula": rdMolDescriptors.CalcMolFormula(mol),
            "Weight": round(Descriptors.MolWt(mol), 2),
            "Atoms": mol.GetNumAtoms(),
            "Chiral Centers": Chem.FindMolChiralCenters(mol, includeUnassigned=True),
        }

        # Detect functional groups
        detected = []
        for name, smarts in FUNCTIONAL_GROUPS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                detected.append(name)
        info["Functional Groups"] = ", ".join(detected) if detected else "None detected"

        # Drug-likeness (Lipinski's Rule of Five)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        donors = Lipinski.NumHDonors(mol)
        acceptors = Lipinski.NumHAcceptors(mol)
        violations = sum([
            mw > 500,
            logp > 5,
            donors > 5,
            acceptors > 10
        ])
        info["Drug-likeness"] = "Yes ✅" if violations <= 1 else "No ❌"

        # Toxic group detection
        toxins = []
        for name, smarts in TOXIC_GROUPS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                toxins.append(name)
        info["Toxic Groups"] = ", ".join(toxins) if toxins else "None"

        return info

    except Exception as e:
        return {"Error": str(e)}
