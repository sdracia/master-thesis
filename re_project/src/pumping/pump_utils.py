import numpy as np
from molecules.molecule import Molecule, CaH, CaOH, CaH_dm2, CaOH_dm2
from QLS.state_dist import States




def validate_laser_configs(laser_configs):
    """
    Validate structure and types of each pulse configuration in laser_configs.
    """

    if not isinstance(laser_configs, (list, tuple)):
        raise TypeError("laser_configs must be a list or tuple of dictionaries.")

    required_keys = {
        "name",
        "is_minus",
        "times",
        "laser_detuning",
        "rabi_rate",
        "raman_config"
    }

    for idx, pulse in enumerate(laser_configs):

        if not isinstance(pulse, dict):
            raise TypeError(f"Pulse {idx} is not a dictionary.")

        # ---- Check required keys ----
        missing_keys = required_keys - pulse.keys()
        if missing_keys:
            raise ValueError(
                f"Pulse {idx} is missing required keys: {missing_keys}"
            )

        # ---- is_minus ----
        if not isinstance(pulse["is_minus"], bool):
            raise TypeError(
                f"Pulse '{pulse['name']}': 'is_minus' must be True or False."
            )

        # ---- times ----
        if not isinstance(pulse["times"], np.ndarray):
            raise TypeError(
                f"Pulse '{pulse['name']}': 'times' must be a numpy array."
            )

        if pulse["times"].ndim != 1:
            raise ValueError(
                f"Pulse '{pulse['name']}': 'times' must be a 1D numpy array."
            )

        if len(pulse["times"]) < 2:
            raise ValueError(
                f"Pulse '{pulse['name']}': 'times' must contain at least 2 elements."
            )

        # ---- laser_detuning ----
        if not isinstance(pulse["laser_detuning"], (float, int)):
            raise TypeError(
                f"Pulse '{pulse['name']}': 'laser_detuning' must be a float."
            )

        # ---- rabi_rate ----
        if not isinstance(pulse["rabi_rate"], (float, int)):
            raise TypeError(
                f"Pulse '{pulse['name']}': 'rabi_rate' must be a float."
            )

        # ---- raman_config ----
        if pulse["raman_config"] not in ("dm1", "dm2"):
            raise ValueError(
                f"Pulse '{pulse['name']}': 'raman_config' must be 'dm1' or 'dm2'."
            )

    return True


def cut_trans_df(transitions_in_j, j_val, keep_sub_manifold_levels):
    rescaled_index = int(np.sum([2*(2*j+1) for j in range(j_val)]))
    df = transitions_in_j.copy()

    m_len = 2*j_val + 1

    if m_len <= keep_sub_manifold_levels:
        return df 
    
    else:
        df["index1"] = df["index1"] - rescaled_index
        df["index2"] = df["index2"] - rescaled_index

        # I keep 5 states on the upper manifold and 5 on the lower one
        df_filtered = df[(df["index1"] < keep_sub_manifold_levels) | ((df["index1"] >= m_len) & (df["index1"] < m_len + keep_sub_manifold_levels))]

        delta = m_len - keep_sub_manifold_levels  

        df_filtered["index1"] = df_filtered["index1"].apply(lambda x: x - delta if x >= m_len else x)
        df_filtered["index2"] = df_filtered["index2"].apply(lambda x: x - delta if x >= m_len else x)


        df_filtered["index1"] = df_filtered["index1"] + rescaled_index
        df_filtered["index2"] = df_filtered["index2"] + rescaled_index

        return df_filtered



def initialize_molecule(molecule_type, b_field_gauss, j_max, temperature):
    """
    Factory function per inizializzare la molecola corretta.
    """

    if molecule_type == "CaH":
        MoleculeClass = CaH
        MoleculeDM2Class = CaH_dm2

    elif molecule_type == "CaOH":
        MoleculeClass = CaOH
        MoleculeDM2Class = CaOH_dm2

    else:
        raise ValueError(
            f"Unsupported molecule_type '{molecule_type}'. "
            "Allowed values are: 'CaH', 'CaOH'."
        )

    mo = MoleculeClass.create_molecule_data(
        b_field_gauss=b_field_gauss,
        j_max=j_max
    )

    states = States(mo, temperature)

    mo_dm2 = MoleculeDM2Class.create_molecule_data_dm2(
        b_field_gauss=b_field_gauss,
        j_max=j_max
    )

    states_dm2 = States(mo_dm2, temperature)

    return mo, states, mo_dm2, states_dm2


