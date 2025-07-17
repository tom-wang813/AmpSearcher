AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Standardized physicochemical properties for 20 amino acids
# Source: Chou, K. C. (2001). Prediction of protein subcellular locations by incorporating
# quasi-amino acid composition into a support vector machine. Proteins: Structure, Function,
# and Bioinformatics, 43(3), 246-255.
# These values are often used in PseAAC calculations.

# Hydrophobicity (H)
HYDROPHOBICITY = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}

# Hydrophilicity (P)
HYDROPHILICITY = {
    "A": -0.50,
    "C": -1.00,
    "D": 3.00,
    "E": 3.00,
    "F": -2.50,
    "G": 0.00,
    "H": -0.50,
    "I": -1.80,
    "K": 3.00,
    "L": -1.80,
    "M": -1.30,
    "N": 0.20,
    "P": 0.00,
    "Q": 0.20,
    "R": 3.00,
    "S": 0.30,
    "T": -0.40,
    "V": -1.50,
    "W": -3.40,
    "Y": -2.30,
}

# Side-chain mass (M)
SIDE_CHAIN_MASS = {
    "A": 15.0,
    "C": 47.0,
    "D": 59.0,
    "E": 73.0,
    "F": 91.0,
    "G": 1.0,
    "H": 81.0,
    "I": 57.0,
    "K": 73.0,
    "L": 57.0,
    "M": 75.0,
    "N": 58.0,
    "P": 42.0,
    "Q": 72.0,
    "R": 100.0,
    "S": 31.0,
    "T": 45.0,
    "V": 43.0,
    "W": 130.0,
    "Y": 107.0,
}

# You can add more properties as needed, e.g., pKa values, charge, etc.

PHYSICOCHEMICAL_PROPERTIES = {
    "H": HYDROPHOBICITY,
    "P": HYDROPHILICITY,
    "M": SIDE_CHAIN_MASS,
}
