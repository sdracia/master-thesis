import json
import numpy as np
from pathlib import Path
from lmfit import Model


def sinc(x, b, c, d):
    sinc_term = np.sinc(b * (x-d) / np.pi) ** 2  # sinc(x) = sin(pi*x)/(pi*x)
    return c * sinc_term


def find_json_in_odf(filename):
    """
    Risale nelle directory finché non trova una cartella chiamata 'odf_json'
    e ritorna il path completo del file richiesto.
    """
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        odf_folder = parent / "odf_json"
        if odf_folder.exists() and odf_folder.is_dir():
            target_file = odf_folder / filename
            if target_file.exists():
                return target_file
            else:
                raise FileNotFoundError(f"{filename} trovato odf_json ma il file non esiste")

    raise FileNotFoundError("Cartella 'odf_json' non trovata risalendo il path")



def odf_data_fitting(max_frequency_mhz, scan_points):
    freq = np.linspace(-max_frequency_mhz, max_frequency_mhz, scan_points)


    json_path = find_json_in_odf("18_01_35_174966.json")

    with open(json_path, "r") as f:
        dfile = json.load(f)

    data = np.array(dfile['data']['mean_excitation']['values'])


    model = Model(sinc)

    params = model.make_params(
        b=2263.33, 
        c=0.4067,  
        d =0.001
    )

    x_data = data[:, 1]  
    y_data = data[:, 0]  

    result = model.fit(y_data, params, x=x_data)

    y_estimated = sinc(freq, **result.best_values)

    return freq, x_data, y_data, result, y_estimated