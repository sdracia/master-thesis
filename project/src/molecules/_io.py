"""
Molecule data input/output routines.

These functions provide class methods to:
- Load molecule state and transition data from CSV files.
- Compute molecule data if files do not exist.
- Save molecule data in a 'molecule_data' directory in the
  current or parent directories.

Separate routines exist for standard and DM2-specific data.
"""

from pathlib import Path
import pandas as pd


def read_molecule_data(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
    """
    Load molecule state and transition data from CSV files.

    Parameters
    ----------
    cls : type
        Molecule class (CaH, CaOH, etc.)
    b_field_gauss : float
        Magnetic field in Gauss
    j_max : int
        Maximum rotational quantum number to consider
    gj_list : list[float], optional
        List of g-factors for different J
    cij_list : list[float], optional
        List of coupling constants for different J

    Returns
    -------
    instance
        Molecule instance with state_df and transition_df populated.
    """
    new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
    states_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_states.csv")
    transitions_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_transitions.csv")

    if states_file.exists() and transitions_file.exists():
        new_instance.state_df = pd.read_csv(states_file)
        new_instance.transition_df = pd.read_csv(transitions_file)
    else:
        print(
            "Molecule data do not exist.\n"
            "Create the molecule using the class method `create_molecule_data`."
        )
    return new_instance


def create_molecule_data(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
    """
    Calculate and save molecule data (states + transitions) given input parameters.

    Parameters
    ----------
    Same as `read_molecule_data`.

    Returns
    -------
    instance
        Molecule instance with state_df and transition_df populated.
    """
    new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
    new_instance.init_states()
    new_instance.init_transition_dataframe()
    new_instance.save_data()
    return new_instance


def read_molecule_data_dm2(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
    """
    Load DM2 molecule data from CSV files (states + DM2-specific transitions).

    Same behavior as `read_molecule_data` but for DM2 variants.
    """
    new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
    states_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_states.csv")
    transitions_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_transitions.csv")

    if states_file.exists() and transitions_file.exists():
        new_instance.state_df = pd.read_csv(states_file)
        new_instance.transition_df = pd.read_csv(transitions_file)
    else:
        print(
            "Molecule DM2 data do not exist.\n"
            "Create the molecule using the class method `create_molecule_data_dm2`."
        )
    return new_instance


def create_molecule_data_dm2(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
    """
    Calculate and save DM2 molecule data (states + DM2 transitions) for given parameters.
    """
    new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
    new_instance.init_states()
    new_instance.init_transition_dm2_dataframe()
    new_instance.save_data()
    return new_instance


def save_data(self):
    """
    Save molecule data (states + transitions) in the nearest 'molecule_data' folder
    found by searching from the current working directory upward.

    Raises
    ------
    FileNotFoundError
        If no 'molecule_data' directory is found in current or parent directories.
    """
    current_path = Path.cwd()

    for parent in [current_path] + list(current_path.parents):
        data_dir = parent / "molecule_data"
        if data_dir.is_dir():
            states_file = data_dir / f"{self.name}_B[{self.b_field_gauss:.2f}]_Jmax[{self.j_max}]_states.csv"
            transitions_file = data_dir / f"{self.name}_B[{self.b_field_gauss:.2f}]_Jmax[{self.j_max}]_transitions.csv"

            self.state_df.to_csv(states_file, index=False)
            self.transition_df.to_csv(transitions_file, index=False)
            print(f"Saved molecule data in:\n{states_file}\n{transitions_file}")
            return

    raise FileNotFoundError("No 'molecule_data' directory found in current or parent directories.")