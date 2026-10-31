
import gsfit_rs  # noqa: F401  (re-exported to the frozen v4 code through stable_helper)
import matplotlib
import numpy as np
from gsfit import Gsfit  # noqa: F401  (re-exported: base.Gsfit)

matplotlib.use("Agg")


def geometry_indices(channels, geometry_channels, flux=False):
    names = [str(name).upper() for name in geometry_channels]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate geometry names")
    indices = {name: index for index, name in enumerate(names)}
    keys = [("FL_" + str(name).replace("/", "_")) if flux else str(name) for name in channels]
    missing = [name for name in keys if name.upper() not in indices]
    if missing:
        raise ValueError(f"Missing geometry: {missing}")
    return [indices[name.upper()] for name in keys]


def psi_to_weber_per_rad(psi):
    return np.asarray(psi) / (2 * np.pi)


def fit_settings(sigma, kind="bp"):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Diagnostic uncertainty must be positive")
    if kind not in ("bp", "flux", "ip"):
        raise ValueError("Unknown diagnostic kind")
    multiplier = 2 * np.pi if kind == "flux" else 1.0
    return {"fit_settings_expected_value": 1.0, "fit_settings_weight": 1.0 / (multiplier * sigma)}


def sensor_role(index, name, stride, excluded):
    if name in excluded:
        return "excluded"
    return "heldout" if stride and index % stride == 0 else "fit"


def add_coil(coils, name, r, z, width, height, value, target):
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    width = np.broadcast_to(np.abs(width), r.shape).astype(float).copy()
    height = np.broadcast_to(np.abs(height), r.shape).astype(float).copy()
    if not np.isfinite(value) or not np.all(np.isfinite(np.concatenate([r, z, width, height]))):
        raise ValueError(f"Invalid coil input: {name}")
    coils.add_pf_coil(name, r, z, width, height, np.array([target]), np.array([value]))


def extract(controller):
    plasma = controller.plasma
    return {
        "r": plasma.get_array1(["grid", "r"]),
        "z": plasma.get_array1(["grid", "z"]),
        "psi": psi_to_weber_per_rad(plasma.get_array3(["profiles_2d", "r_z", "psi"])[0]),
        "gs_error": float(plasma.get_array1(["global", "gs_error"])[0]),
        "iterations": int(plasma.get_vec_usize(["global", "n_iter"])[0]),
        "ip": float(plasma.get_array1(["global", "ip"])[0]),
        "axis_r": float(plasma.get_array1(["global", "r_mag"])[0]),
        "axis_z": float(plasma.get_array1(["global", "z_mag"])[0]),
        "boundary_r": plasma.get_array2(["boundary", "outline", "r"])[0],
        "boundary_z": plasma.get_array2(["boundary", "outline", "z"])[0],
        "q_95": float(plasma.get_array1(["global", "q_95"])[0]),
        "q_axis": float(plasma.get_array1(["global", "q_axis"])[0]),
        "beta_n": float(plasma.get_array1(["global", "beta_n"])[0]),
        "beta_p_1": float(plasma.get_array1(["global", "beta_p_1"])[0]),
        "beta_t": float(plasma.get_array1(["global", "beta_t"])[0]),
        "li_1": float(plasma.get_array1(["global", "li_1"])[0]),
        "w_mhd": float(plasma.get_array1(["global", "w_mhd"])[0]),
        "bt_vac_at_r_geo": float(plasma.get_array1(["global", "bt_vac_at_r_geo"])[0]),
        "i_rod": float(plasma.get_array1(["global", "i_rod"])[0]),
    }
