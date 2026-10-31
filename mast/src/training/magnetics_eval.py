"""指定线圈电流、汤姆逊与 Ip 输入的磁诊断评估；真值仅参与评分。"""

from __future__ import annotations



from .magnetics_inputs import (
    sanitize_inputs,
)



def predict(model, batch, specs, selected, device):
    values, masks = sanitize_inputs(batch, specs, selected, model.patches, device)
    t0 = batch['t0'].to(device)
    # 查询固定为全部位置，不以目标真值是否存在决定查询。
    reconstruction, visible = model.reconstruct(values, masks, t0)
    return reconstruction, visible.any(dim=(1, 2))
