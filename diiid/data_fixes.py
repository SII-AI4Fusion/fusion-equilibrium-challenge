"""Drop-in fixes for the two known data errata in dataset release v1.1.0.

Both issues were participant-reported (thank you!) and affect *input* columns
only -- the flux-map targets, ``efit_times``, and all scoring are unaffected,
and every participant has the same data. A corrected v1.1.1 dataset release
will match these fixes exactly, so anything you build on top of them carries
over unchanged.

1.  **DIII-D ``magnetics_plasma_current_times`` is wrong on ~69% of shots**
    (`issue #6 <https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/6>`_).
    Every DIII-D file shipped with the *same* Ip time axis. That axis is
    genuinely correct for ~31% of shots, but on the rest the Ip trace actually
    starts at ``magnetics_time[0]`` -- about three seconds earlier -- so
    interpolating Ip onto ``efit_times`` with the shipped axis reads the
    pre-plasma noise floor (~2 kA) instead of the real current (~1 MA).
    The Ip *values* are correct on every shot; only the timestamps are wrong.
    Use :func:`fix_d3d_ip_times`. MAST is unaffected (its Ip sits on
    ``magnetics_time``).

2.  **MAST ``thomson_core_R`` has 130 entries but Te/ne have 131 channels**
    (`issue #5 <https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/5>`_).
    Channel 0 of the raw Thomson system has no radius calibration and is NaN
    in every shot; its coordinate was dropped at conversion time but its
    (empty) data column was kept. The alignment is therefore
    ``R[j] <-> Te[t][j + 1]``. Use :func:`align_mast_thomson_core`.

Run ``python data_fixes.py`` to see both fixes applied to the demo shots in
``parquet_data/`` (requires ``git lfs pull``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["fix_d3d_ip_times"]

# On DIII-D, t = 0 is plasma breakdown by convention, so the correct Ip axis
# is the one that puts the first significant current near t = 0. Across all
# 9,113 DIII-D source shots the winning axis puts it within [-118, +348] ms
# while the losing axis lands seconds away, so this window is unambiguous.
_BREAKDOWN_WINDOW_MS = (-500.0, 1000.0)


def fix_d3d_ip_times(row) -> np.ndarray:
    """Return the corrected time axis for ``magnetics_plasma_current`` (DIII-D).

    Accepts a Hugging Face dataset row (dict) or a pandas row from a demo
    parquet. It needs only input columns, so it works identically on the
    train and the inputs-only test splits.

    Per shot, this keeps the shipped ``magnetics_plasma_current_times`` if it
    already places plasma breakdown (first |Ip| crossing of 10% of the shot's
    peak) near t = 0, and otherwise re-origins the axis to start at
    ``magnetics_time[0]``, which is where the affected shots' Ip acquisition
    actually began. Validated on all 9,113 DIII-D source shots: the decision
    is unambiguous on every one, correcting 6,322 shots (4,930 of 7,041 in
    ``diii_d_train``, 586 of 874 in ``diii_d_public_test``) and leaving the
    already-correct 2,791 untouched.

    Do NOT blanket-shift every shot instead -- ~31% of shots ship with the
    correct axis, and shifting those would corrupt them.
    """
    ipt = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    ip = np.abs(np.asarray(row["magnetics_plasma_current"], dtype=np.float64))
    threshold = 0.10 * np.nanpercentile(ip, 99.5)
    breakdown = float(ipt[int(np.argmax(ip > threshold))])
    lo, hi = _BREAKDOWN_WINDOW_MS
    if lo <= breakdown <= hi:
        return ipt
    return float(row["magnetics_time"][0]) + (ipt - ipt[0])
