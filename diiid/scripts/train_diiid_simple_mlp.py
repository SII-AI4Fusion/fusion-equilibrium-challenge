"""Starter-kit DIII-D baseline at full scale: SimpleMLP (raw 65x65 pixels, no PCA) + Ridge q95/betaN.

Model and optimiser follow experiments.py / experiments_torch.py defaults: the 21 coil/Ip traces
interpolated onto efit_times (Ip axis fixed), StandardScaler, SimpleMLP 256-512-1024-4225 with
BatchNorm, MSE on raw psi, Adam 1e-3, batch 256, 50 epochs, ReduceLROnPlateau(5, 0.5), last epoch
kept; q95/betaN by RidgeCV(alphas=logspace(-3,3,10)) on the same scaled features (GCV, as sklearn).
Features and labels are read from the cache written by build_diiid_cache.py (--pca-run), whose
seed-42 shot split (80/10/10) is reused.

    train   --pca-run DIR --out DIR             fit, write model.pt and test_metrics.json
    predict --out DIR --input-dir DIR --output F.npz   (psirz written as float16, as submitted)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
import build_diiid_cache as pca_training  # noqa: E402

SIGNALS = pca_training.SIGNALS
GRID = 65
ALPHAS = np.logspace(-3, 3, 10)


class SimpleMLP(nn.Module):
    """experiments_torch.SimpleMLP, unchanged."""

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, GRID * GRID),
        )

    def forward(self, x):
        return self.net(x)


def save_json(path, obj):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n'); tmp.replace(path)


def feature_index(pca_run):
    names = json.loads((pca_run / 'feature_layout.json').read_text())['feature_names']
    return np.array([names.index(f'magnetics_{s}') for s in SIGNALS])


def load_split(pca_run, files, idx, workers=16):
    def one(f):
        with np.load(pca_run / 'cache' / (Path(f).stem + '.npz')) as z:
            return z['x'][:, idx].astype(np.float32), z['y'].astype(np.float32), z['s'].astype(np.float32)
    with ThreadPoolExecutor(workers) as pool:
        parts = list(pool.map(one, files))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def ridge_cv(X, y):
    """RidgeCV with intercept and efficient leave-one-out (GCV mode) over ALPHAS."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uty = U.T @ yc
    best = None
    for a in ALPHAS:
        f = s ** 2 / (s ** 2 + a)
        h = (U ** 2) @ f + 1.0 / len(y)
        resid = yc - U @ (f * Uty)
        loo = float(np.mean((resid / (1 - h)) ** 2))
        if best is None or loo < best[0]:
            best = (loo, a, Vt.T @ (Uty * s / (s ** 2 + a)))
    loo, a, coef = best
    return {'coef': coef, 'intercept': float(ym - xm @ coef), 'alpha': float(a), 'loo_mse': loo}


def r2(t, p):
    t, p = np.asarray(t, np.float64), np.asarray(p, np.float64)
    return float(1 - np.square(t - p).sum() / np.square(t - t.mean()).sum())


def predict_arrays(model, scaler, ridges, X, device):
    Xs = ((X - scaler['mean']) / scaler['scale']).astype(np.float32)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            out.append(model(torch.from_numpy(Xs[i:i + 8192]).to(device)).float().cpu().numpy())
    psi = np.concatenate(out) if out else np.zeros((0, GRID * GRID), np.float32)
    scal = np.column_stack([Xs.astype(np.float64) @ r['coef'] + r['intercept'] for r in ridges]).astype(np.float32)
    return psi.reshape(-1, GRID, GRID), scal


def load_model(out, device):
    ck = torch.load(out / 'model.pt', map_location=device, weights_only=False)
    model = SimpleMLP(len(SIGNALS)).to(device); model.load_state_dict(ck['state_dict'])
    return model, ck['scaler'], ck['ridges']


def cmd_train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    splits = json.loads((a.pca_run / 'splits.json').read_text())
    if a.limit_shots:
        splits = {k: v[:max(2, a.limit_shots * len(v) // 7041)] for k, v in splits.items()}
    idx = feature_index(a.pca_run)
    t0 = time.time()
    Xtr, Ytr, Str = load_split(a.pca_run, splits['train'], idx)
    Xva, Yva, _ = load_split(a.pca_run, splits['val'], idx)
    print(f'loaded train {Xtr.shape} val {Xva.shape} in {time.time() - t0:.0f}s', flush=True)
    mean = Xtr.mean(0, dtype=np.float64); std = Xtr.std(0, dtype=np.float64); std[std == 0] = 1.0
    scaler = {'mean': mean, 'scale': std}
    sc = lambda X: ((X - mean) / std).astype(np.float32)  # noqa: E731

    ridges = []
    for j, name in enumerate(['q95', 'betaN']):
        m = np.isfinite(Str[:, j])
        r = ridge_cv(sc(Xtr[m]).astype(np.float64), Str[m, j].astype(np.float64)); r['name'] = name
        ridges.append(r); print(f'ridge {name}: alpha={r["alpha"]:.3g} loo_mse={r["loo_mse"]:.4g}', flush=True)

    xt = torch.from_numpy(sc(Xtr)).to(device); yt = torch.from_numpy(Ytr.reshape(len(Ytr), -1)).to(device)
    xv = torch.from_numpy(sc(Xva)).to(device); yv = torch.from_numpy(Yva.reshape(len(Yva), -1)).to(device)
    del Xtr, Ytr, Xva, Yva
    model = SimpleMLP(xt.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss(); history = []
    for epoch in range(a.epochs):
        model.train(); perm = torch.randperm(len(xt), device=device); tot = 0.0; nb = 0
        for i in range(0, len(xt), a.batch_size):
            b = perm[i:i + a.batch_size]
            if len(b) < 2:
                continue  # BatchNorm needs >1 sample; DataLoader would pass it and fail
            loss = loss_fn(model(xt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        model.eval(); vl = 0.0; nvb = 0
        with torch.no_grad():
            for i in range(0, len(xv), a.batch_size * 16):
                vl += loss_fn(model(xv[i:i + a.batch_size * 16]), yv[i:i + a.batch_size * 16]).item(); nvb += 1
        vl /= max(nvb, 1); sched.step(vl)
        history.append({'epoch': epoch + 1, 'train': tot / max(nb, 1), 'val': vl, 'lr': opt.param_groups[0]['lr'], 't': time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
    torch.save({'state_dict': model.state_dict(), 'scaler': scaler, 'ridges': ridges, 'signals': SIGNALS,
                'config': {k: str(v) for k, v in vars(a).items() if k != 'func'}}, a.out / 'model.pt')
    save_json(a.out / 'history.json', history)
    save_json(a.out / 'splits_used.json', {k: len(v) for k, v in splits.items()})

    del xt, yt, xv, yv
    Xte, Yte, Ste = load_split(a.pca_run, splits['test'], idx)
    psi, scal = predict_arrays(model, scaler, ridges, Xte, device)
    metrics = {'shots': len(splits['test']), 'frames': len(Xte),
               'psi_r2_flattened': r2(Yte.reshape(-1), psi.reshape(-1)),
               'psi_rmse': float(np.sqrt(np.mean(np.square(Yte.reshape(len(Yte), -1) - psi.reshape(len(psi), -1))))),
               'elapsed_seconds': time.time() - t0, 'device': str(device)}
    for j, name in enumerate(['q95', 'betaN']):
        m = np.isfinite(Ste[:, j]); metrics[name] = {'r2': r2(Ste[m, j], scal[m, j]), 'alpha': ridges[j]['alpha']}
    save_json(a.out / 'test_metrics.json', metrics); print(json.dumps(metrics), flush=True)


def shot_features(path):
    cols = ['efit_times', 'magnetics_time', 'magnetics_plasma_current_times'] + ['magnetics_' + s for s in SIGNALS]
    t = pca_training.pq.read_table(path, columns=cols, use_threads=False)
    row = {k: pca_training.numeric(t, k) for k in cols}
    times = row['efit_times']; ip_times = pca_training.fix_d3d_ip_times(row)
    x = np.column_stack([pca_training.interpolation(ip_times if s == 'plasma_current' else row['magnetics_time'],
                                                    row['magnetics_' + s], times, True)[0] for s in SIGNALS])
    return x.astype(np.float32), len(times)


def cmd_predict(a):
    device = torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    model, scaler, ridges = load_model(a.out, device)
    files = sorted(a.input_dir.glob('*.parquet'))
    if a.max_shots:
        files = files[:a.max_shots]
    sub, rows = {}, []
    for i, f in enumerate(files):
        x, n = shot_features(f)
        if not np.isfinite(x).all():
            raise ValueError(f'nonfinite input {f}')
        psi, scal = predict_arrays(model, scaler, ridges, x, device)
        assert psi.shape == (n, GRID, GRID) and np.isfinite(psi).all() and np.isfinite(scal).all(), f
        sub[f'shot_{i:04d}_psirz'] = psi.astype(np.float32).astype(a.psi_dtype)
        sub[f'shot_{i:04d}_q95'] = scal[:, 0]; sub[f'shot_{i:04d}_betaN'] = scal[:, 1]
        rows.append({'index': i, 'file': f.name, 'frames': n})
        if (i + 1) % 100 == 0:
            print('predicted', i + 1, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(a.output.stem + '.partial.npz'); np.savez(tmp, **sub); tmp.replace(a.output)
    save_json(a.output.with_suffix('.manifest.json'), {'model': str(a.out / 'model.pt'), 'input_dir': str(a.input_dir),
              'order': 'sorted parquet filenames (= HF data_files order)', 'label_columns_read': [], 'shots': rows})
    print(a.output, len(files), 'shots', flush=True)


def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(required=True)
    t = sp.add_parser('train'); t.set_defaults(func=cmd_train)
    t.add_argument('--pca-run', type=Path, required=True); t.add_argument('--out', type=Path, required=True)
    t.add_argument('--epochs', type=int, default=50); t.add_argument('--batch-size', type=int, default=256)
    t.add_argument('--lr', type=float, default=1e-3); t.add_argument('--seed', type=int, default=42)
    t.add_argument('--limit-shots', type=int, default=0); t.add_argument('--device', default='cuda')
    q = sp.add_parser('predict'); q.set_defaults(func=cmd_predict)
    q.add_argument('--out', type=Path, required=True); q.add_argument('--input-dir', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True); q.add_argument('--device', default='cuda')
    q.add_argument('--max-shots', type=int, default=0)
    q.add_argument('--psi-dtype', choices=['float16', 'float32'], default='float16',
                   help='dtype of the submitted psirz arrays (the Final submission used float16)')
    a = p.parse_args(); a.func(a)


if __name__ == '__main__':
    main()
