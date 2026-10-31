from __future__ import annotations

import re


SIGNALS = {
    "magnetics/flux_loop_flux": ("flux_loop_channel", "flux"),
    "magnetics/b_field_pol_probe_ccbv_field": ("b_field_pol_probe_ccbv_channel", "bp"),
    "magnetics/b_field_pol_probe_obr_field": ("b_field_pol_probe_obr_channel", "bp"),
    "magnetics/b_field_pol_probe_obv_field": ("b_field_pol_probe_obv_channel", "bp"),
}
PROTOCOL = dict(
    dof=1,
    regularisation=1.0,
    initial_r=0.9,
    minor_radius=0.35,
    kappa=1.5,
    max_iter=100,
    bp_sigma=0.002,
    flux_sigma=0.005,
    passives="measured",
    holdout_stride=0,
    exclude_sensor=["P4L/1", "P4L/4"],
)


def canonical_sensor_name(name):
    name = str(name).strip().upper()
    if name.startswith("AMB_FL/"):
        return name[len("AMB_FL/") :]
    if name.startswith("AMB_"):
        return name[len("AMB_") :]
    return name


def physical_channel_map(channels, labels):
    out = []
    for column, channel in enumerate(channels):
        match = re.fullmatch(r"(.+)\[(\d+)\]", channel)
        if not match or match[1] not in SIGNALS:
            continue
        signal, index = match[1], int(match[2])
        if index >= len(labels.get(signal, [])):
            continue
        out.append(
            dict(
                column=column,
                name=canonical_sensor_name(labels[signal][index]),
                source_name=str(labels[signal][index]),
                signal=signal,
                kind=SIGNALS[signal][1],
            )
        )
    names = [row["name"] for row in out]
    if len(names) != len(set(names)):
        raise ValueError("Ambiguous magnetic diagnostic names")
    return out
