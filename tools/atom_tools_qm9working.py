import ase
import torch


def atom_counts_to_vector(formula: str):
    atom_counts_dict = parse_formula(formula)
    # Define atomic types
    atomic_types = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
                    'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
                    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                    'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
                    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                    'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                    'No', 'Lr']

    # Initialize torch tensor with zeros
    atom_vector = torch.zeros(119, dtype=torch.int)

    # Fill in the counts of each atom type
    for atomic_symbol, count in atom_counts_dict.items():
        atomic_index = atomic_types.index(atomic_symbol)
        atom_vector[atomic_index] = count

    return atom_vector


def parse_formula(formula):
    # Parses a molecule into a dictionary of element counts
    elements = {}
    current_element = ""
    current_count = ""
    for char in formula:
        if char.isupper():
            if current_element:
                elements[current_element] = int(current_count) if current_count else 1
            current_element = char
            current_count = ""
        elif char.islower():
            current_element += char
        elif char.isdigit():
            current_count += char
    if current_element:
        elements[current_element] = int(current_count) if current_count else 1
    return elements


def valence_electrons(chemical_formula):
    valence_dict = {
        'X': 0,
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 4, 'N': 5, 'O': 6,
        'F': 7, 'Ne': 8, 'Na': 9, 'Mg': 10, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29}

    form_dict = parse_formula(chemical_formula)
    valence_electrons = 0
    for key, value in form_dict.items():
        valence_electrons += valence_dict[key] * value

    return valence_electrons

def main():
    # Example usage
    atom_vector = atom_counts_to_vector("C6H6")
    print(atom_vector)


if __name__ == "__main__":
    main()
