"""磁诊断推理与有监督后训练共用的输入白名单与数据消毒逻辑。

与 ``magnetics_eval`` / GSFit 推理一致：仅开放 ``summary/ip``、``pf_active`` 线圈与螺线管电流、
以及模型中存在的 Thomson 诊断（``thomson_core`` / ``thomson_profiles``）；目标真值仅用于计分，
绝不进入编码器。``E_pf_tf_ts_ip`` 另开放 ``tf/coil_current``，对应比赛允许的全部时间信号输入。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from model.router import ModalitySpec

EXPERIMENTS = ("A_ip", "B_pf_ip", "C_ts_ip", "D_pf_ts_ip", "D_shuffled")
# 有监督训练可用的输入集；E 需要 v3 缓存的 tf 模态，不在 magnetics_eval 的 v2 评估集合中。
SUPERVISED_INPUTS = (*EXPERIMENTS, "E_pf_tf_ts_ip")


def input_channels(layout: dict[str, Any], experiment: str) -> dict[str, list[int]]:
    """白名单精确到缓存通道；线圈电压和其余 summary 通道不开放。"""
    selected: dict[str, list[int]] = {}
    by_name = {m["name"]: m for m in layout["modalities"]}
    selected["summary"] = [by_name["summary"]["channels"].index("summary/ip")]
    if experiment in ("B_pf_ip", "D_pf_ts_ip", "D_shuffled", "E_pf_tf_ts_ip"):
        selected["pf_active"] = [
            i for i, n in enumerate(by_name["pf_active"]["channels"])
            if n.startswith("pf_active/coil_current[") or n == "pf_active/solenoid_current"
        ]
        if not selected["pf_active"]:
            raise ValueError("没有找到线圈电流通道")
    if experiment in ("C_ts_ip", "D_pf_ts_ip", "D_shuffled", "E_pf_tf_ts_ip"):
        for name in ("thomson_core", "thomson_profiles"):
            if name in by_name:
                selected[name] = list(range(len(by_name[name]["channels"])))
    if experiment == "E_pf_tf_ts_ip":
        if "tf" not in by_name:
            raise ValueError("E_pf_tf_ts_ip 需要含 tf 模态的缓存（FAIR-MAST v3）")
        selected["tf"] = [by_name["tf"]["channels"].index("tf/coil_current")]
    return selected


def target_channels(layout: dict[str, Any], name: str) -> list[int]:
    m = next(m for m in layout["modalities"] if m["name"] == name)
    return [i for i, n in enumerate(m["channels"]) if n != "magnetics/ip"]


def sanitize_inputs(
    batch: dict[str, Any],
    specs: list[ModalitySpec],
    selected: dict[str, list[int]],
    patches: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """忽略所有非白名单数值及缺测信息，不读取隐藏目标生成 availability。"""
    b = len(batch["t0"])
    values, masks = {}, {}
    for spec in specs:
        shape = (b, spec.channels, *spec.spatial, patches * spec.samples_per_patch)
        values[spec.name] = torch.zeros(shape, device=device)
        masks[spec.name] = torch.zeros(shape, device=device, dtype=torch.bool)
        cols = selected.get(spec.name, [])
        if cols:
            observed = batch["masks"][spec.name][:, cols].to(device)
            raw = batch["inputs"][spec.name][:, cols].to(device)
            values[spec.name][:, cols] = torch.where(observed, raw, 0)
            masks[spec.name][:, cols] = observed
    return values, masks


def prepare_supervised_batch(
    batch: dict[str, Any],
    specs: list[ModalitySpec],
    layout: dict[str, Any],
    experiment: str,
    targets: list[str],
    patches: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
    """仅向编码器提供白名单输入，保留目标模态真值供解码端计算损失。

    返回 ``(values, masks, hidden)``：白名单输入处 ``hidden=False``（进入编码器），
    目标模态有观测处 ``hidden=True``（被遮挡并参与损失计分），其余模态掩码恒为 ``False``。
    目标中的 ``magnetics/ip`` 从损失掩码中剔除，与评估协议保持一致。
    """
    selected = input_channels(layout, experiment)
    values, masks = sanitize_inputs(batch, specs, selected, patches, device)
    name_to_idx = {spec.name: i for i, spec in enumerate(specs)}
    for name in targets:
        if name not in name_to_idx:
            raise KeyError(f"supervised target {name!r} is not in the model modalities")
        if name in selected:
            raise ValueError(f"supervised target {name!r} overlaps with input {experiment!r}")
        values[name] = batch["inputs"][name].to(device)
        target_mask = batch["masks"][name].to(device).clone()
        keep = set(target_channels(layout, name))
        drop = [i for i in range(target_mask.shape[1]) if i not in keep]
        if drop:
            target_mask[:, drop] = False
        masks[name] = target_mask

    coverage = []
    for spec in specs:
        mask = masks[spec.name].float()
        per_patch = mask.reshape(mask.shape[0], -1, patches, spec.samples_per_patch)
        coverage.append(per_patch.mean(dim=(1, 3)))
    available = torch.stack(coverage, dim=1) > 0
    hidden = available.clone()
    for mod_name in selected:
        if mod_name in name_to_idx:
            hidden[:, name_to_idx[mod_name]] = False
    return values, masks, hidden
