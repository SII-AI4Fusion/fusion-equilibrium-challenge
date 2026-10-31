"""Frozen GSFit integration helper with independent stable scalar extraction."""
import importlib.util
from pathlib import Path

import numpy as np
from training.gsfit_postprocess import stable_scalars

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("original_gsfit_helper", HERE / "assets/fair_mast_reconstruct_v2.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
for name in dir(base):
    if not name.startswith("_"):
        globals()[name] = getattr(base, name)
original_extract = base.extract


def extract(controller):
    result = original_extract(controller)
    plasma = controller.plasma
    result["native_q95"], result["native_betaN"] = result["q_95"], result["beta_n"]
    result["psi_axis"] = float(plasma.get_array1(["global", "psi_a"])[0]) / (2 * np.pi)
    result["psi_boundary"] = float(plasma.get_array1(["boundary", "psi"])[0]) / (2 * np.pi)
    result["F_profile"] = plasma.get_array2(["profiles_1d", "psi_norm", "f"])[0]
    result["p_profile"] = plasma.get_array2(["profiles_1d", "psi_norm", "p"])[0]
    values = stable_scalars(result["psi"], result["r"], result["z"], result["psi_axis"], result["psi_boundary"],
                            result["F_profile"], result["p_profile"], result["ip"], result["i_rod"],
                            result["axis_r"], result["axis_z"],
                            vessel=np.column_stack([plasma.get_array1(["vessel","r"]), plasma.get_array1(["vessel","z"])]))
    result.update(q_95=values["q95"], beta_n=values["betaN"], postprocess_volume=values["volume"],
                  boundary_level=values["boundary_level"], q95_error=values["q95_error"], betaN_error=values["betaN_error"])
    return result

original_fit_settings = fit_settings  # noqa: F821  (copied from the base module above)
def fit_settings(sigma, kind="bp"):
    return original_fit_settings(500. if kind=="ip" else sigma, kind)
