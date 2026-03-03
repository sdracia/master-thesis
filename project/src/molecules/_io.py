from pathlib import Path
import pandas as pd


def read_molecule_data(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
    """
    Load the molecule data from the file. If the file does not exist, returns an error.
    """
    new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
    states_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_states.csv")
    transitions_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_transitions.csv")


    if states_file.exists() and transitions_file.exists():
        new_instance.state_df = pd.read_csv(states_file)
        new_instance.transition_df = pd.read_csv(transitions_file)
    else:
        print("The molecule data do not exist.'\n' Create the new molecule with class method create_molecule_data")
    return new_instance

def create_molecule_data(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
    """
    Calculate the molecule data given the input parameters.
    """
    new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
    new_instance.init_states()
    new_instance.init_transition_dataframe()
    new_instance.save_data()
    return new_instance




def read_molecule_data_dm2(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
    """
    Load the molecule data from the file. If the file does not exist, returns an error.
    """
    new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
    states_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_states.csv")
    transitions_file = Path(f"molecule_data/{cls.name}_B[{b_field_gauss:.2f}]_Jmax[{j_max}]_transitions.csv")


    if states_file.exists() and transitions_file.exists():
        new_instance.state_df = pd.read_csv(states_file)
        new_instance.transition_df = pd.read_csv(transitions_file)
    else:
        print("The molecule data do not exist.'\n' Create the new molecule with class method create_molecule_data")
    return new_instance


def create_molecule_data_dm2(cls, b_field_gauss: float, j_max: int, gj_list: list[float] = None, cij_list: list[float] = None):
    """
    Calculate the molecule data given the input parameters.
    """
    new_instance = cls(b_field_gauss, j_max, gj_list, cij_list)
    new_instance.init_states()
    new_instance.init_transition_dm2_dataframe()
    new_instance.save_data()
    return new_instance





# def save_data(self):
#     self.state_df.to_csv(f"molecule_data/{self.name}_B[{self.b_field_gauss:.2f}]_Jmax[{self.j_max}]_states.csv", index=False)
#     self.transition_df.to_csv(f"molecule_data/{self.name}_B[{self.b_field_gauss:.2f}]_Jmax[{self.j_max}]_transitions.csv", index=False)




def save_data(self):
    """
    Save the molecule data (states + transitions) inside the nearest 'molecule_data' folder 
    found by going up the directory tree from the current working directory.
    """
    current_path = Path.cwd()  # directory da cui è lanciato il notebook
    
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