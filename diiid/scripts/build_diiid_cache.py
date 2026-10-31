"""DIII-D feature cache for the SimpleMLP baseline: one NPZ per diii_d_train shot plus the shot split.

This is the cache stage of the original PCA-MLP baseline script (its PCA/MLP training is not used).
Per shot it stores the engineered inputs ``x`` (21 coil/Ip traces with the official Ip time fix,
Thomson channels and geometry, interpolated onto efit_times), the psi targets ``y`` and the
q95/betaN scalars ``s``; frames with non-finite psi are dropped. Splits are by shot:
RandomState(42) permutation, 10 % val, 10 % test.

    python build_diiid_cache.py --data <challenge dataset root> --out <dir> [--workers 8]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_fixes import fix_d3d_ip_times

SIGNALS = ['ECOILA', 'bcoil'] + [f'F{i}{side}' for i in range(1, 10) for side in 'AB'] + ['plasma_current']
LABELS = ['efit_psirz', 'efit_q95', 'efit_beta_n']
FORBIDDEN = ['magnetics_dsep', 'magnetics_dsep_times', 'efit_li', 'efit_r_axis', 'efit_z_axis', 'efit_lcfs_n', 'efit_lcfs_r', 'efit_lcfs_z']
STATIC = ['thomson_core_R', 'thomson_edge_spatial', 'coil_R', 'coil_Z', 'coil_width', 'coil_height', 'coil_angle1', 'coil_angle2', 'thomson_chord_R', 'thomson_chord_Z', 'efit_grid_R', 'efit_grid_Z']
META = ['source', 'coil_name', 'coil_input_column', 'thomson_chord_name']
INPUTS = ['efit_times', 'magnetics_time', 'magnetics_plasma_current_times'] + ['magnetics_' + s for s in SIGNALS] + [f'thomson_{system}_{v}' for system in ['core', 'edge'] for v in ['times', 'Te', 'ne']] + STATIC + META
CHANNEL_CAPACITY = {'core':64, 'edge':32}
CHORD_IDS = [f'TS_core_{i}' for i in range(64)] + [f'TS_tangential_{i}' for i in range(32)] + [f'TS_divertor_{i}' for i in range(32)]


def geometry_summary(path):
    t=pq.read_table(path,columns=['thomson_core_R','thomson_edge_spatial','thomson_chord_name'],use_threads=False)
    chords=[v.split('_r')[0]+'_'+v.rsplit('_',1)[-1] for v in t['thomson_chord_name'][0].as_py()]
    return len(t['thomson_core_R'][0].values),len(t['thomson_edge_spatial'][0].values),chords


def configure_layout(files,workers):
    global CHORD_IDS, CHANNEL_CAPACITY
    chords=set();core=edge=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for a,b,c in pool.map(geometry_summary,files):
            core=max(core,a);edge=max(edge,b);chords.update(c)
    CHANNEL_CAPACITY={'core':core,'edge':edge}
    CHORD_IDS=sorted(chords,key=lambda v:(v.rsplit('_',1)[0],int(v.rsplit('_',1)[1])))


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def numeric(table, name):
    a = table[name].combine_chunks()[0].values
    shape = [len(a)]
    while pa.types.is_list(a.type) or pa.types.is_large_list(a.type):
        sizes = np.asarray(a.value_lengths())
        if len(sizes) and not np.all(sizes == sizes[0]):
            raise ValueError(f'ragged {name}')
        shape.append(int(sizes[0]) if len(sizes) else 0)
        a = a.flatten()
    return a.to_numpy(zero_copy_only=False).astype(np.float64, copy=False).reshape(shape)


def interpolation(t, v, target, extrapolate=False):
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < 2:
        return np.zeros(len(target)), np.zeros(len(target))
    idx = np.argsort(t, kind='stable'); t, v = t[idx], v[idx]
    t, idx = np.unique(t, return_index=True); v = v[idx]
    if len(t) < 2:
        return np.zeros(len(target)), np.zeros(len(target))
    out = np.interp(target, t, v)
    valid = (target >= t[0]) & (target <= t[-1])
    if extrapolate:
        left, right = target < t[0], target > t[-1]
        out[left] = v[0] + (target[left] - t[0]) * (v[1] - v[0]) / (t[1] - t[0])
        out[right] = v[-1] + (target[right] - t[-1]) * (v[-1] - v[-2]) / (t[-1] - t[-2])
    else:
        out[~valid] = 0
    return out, valid.astype(float)


def read_case(path, layout=None, labels=True):
    cols = INPUTS + (LABELS if labels else [])
    table = pq.read_table(path, columns=cols, use_threads=False)
    row = {k: numeric(table, k) for k in cols if k not in META}
    meta = {k: table[k][0].as_py() for k in META}
    times = row['efit_times']; n = len(times)
    iptimes = fix_d3d_ip_times(row)
    parts, names = [times[:, None] / 1000], ['efit_time_seconds']
    for sig in SIGNALS:
        v, mask = interpolation(iptimes if sig == 'plasma_current' else row['magnetics_time'], row['magnetics_' + sig], times, True)
        parts.extend([v[:, None], mask[:, None]])
        names.extend([f'magnetics_{sig}', f'magnetics_{sig}:in_time_range'])
    for system in ['core', 'edge']:
        for measure in ['Te', 'ne']:
            k = f'thomson_{system}_{measure}'
            v = row[k]
            if v.shape[1] > CHANNEL_CAPACITY[system]:
                raise ValueError(f'Thomson capacity exceeded: {path}')
            for ch in range(CHANNEL_CAPACITY[system]):
                values, mask = interpolation(row[f'thomson_{system}_times'], v[:, ch], times) if ch < v.shape[1] else (np.zeros(n),np.zeros(n))
                parts.extend([values[:, None], mask[:, None]])
                names.extend([f'{k}[{ch}]', f'{k}[{ch}]:observed'])
    chord_ids = [v.split('_r')[0] + '_' + v.rsplit('_', 1)[-1] for v in meta['thomson_chord_name']]
    coil_ids = [v.removeprefix('magnetics_') for v in meta['coil_input_column']]
    if len(set(chord_ids)) != len(chord_ids) or not set(chord_ids).issubset(CHORD_IDS) or not set(coil_ids).issubset(SIGNALS):
        raise ValueError(f'unknown or duplicate channel identity: {path}')
    for k in STATIC:
        raw = row[k].reshape(-1)
        if k.startswith('thomson_chord_'):
            if len(raw)!=len(chord_ids):raise ValueError(f'chord geometry mismatch: {path}')
            a=np.full(len(CHORD_IDS),np.nan)
            for name,value in zip(chord_ids,raw):a[CHORD_IDS.index(name)]=value
        elif k.startswith('coil_'):
            if len(raw)!=len(coil_ids):raise ValueError(f'coil geometry mismatch: {path}')
            a=np.full(len(SIGNALS),np.nan)
            for name,value in zip(coil_ids,raw):a[SIGNALS.index(name)]=value
        elif k in ['thomson_core_R','thomson_edge_spatial']:
            capacity=CHANNEL_CAPACITY['core' if k=='thomson_core_R' else 'edge']
            if len(raw)>capacity:raise ValueError(f'spatial capacity exceeded: {path}')
            a=np.full(capacity,np.nan);a[:len(raw)]=raw
        else:
            a=raw
        good = np.isfinite(a)
        parts.extend([np.broadcast_to(np.where(good, a, 0), (n, len(a))), np.broadcast_to(good, (n, len(a)))])
        names.extend([f'{k}[{i}]' for i in range(len(a))] + [f'{k}[{i}]:observed' for i in range(len(a))])
    x = np.concatenate(parts, axis=1).astype(np.float32)
    # The chord name embeds a rounded R coordinate and changes between shots;
    # preserve system/channel identity, with its actual R/Z supplied numerically.
    contract = {'feature_names': names, 'coil_input_column': SIGNALS, 'thomson_chord_name': CHORD_IDS,'channel_capacity':CHANNEL_CAPACITY}
    if layout is not None and contract != layout:
        raise ValueError(f'input layout changed: {path}')
    if not np.isfinite(x).all():
        raise ValueError(f'nonfinite engineered input: {path}')
    result = {'x': x, 'times': times, 'ip_axis_corrected': bool(not np.array_equal(iptimes, row['magnetics_plasma_current_times']))}
    if labels:
        y = row['efit_psirz'].reshape(n, -1).astype(np.float32)
        if y.shape[1] != 4225:
            raise ValueError(f'wrong psi grid: {y.shape}')
        scalars = np.column_stack([row['efit_q95'], row['efit_beta_n']]).astype(np.float32)
        valid = np.isfinite(y).all(axis=1)
        result.update(y=y[valid], x=x[valid], s=scalars[valid], times=times[valid], dropped=int((~valid).sum()))
    return result, contract


def cache_case(item):
    source, dest, layout = item
    dest = Path(dest)
    if dest.exists():
        with np.load(dest) as z:
            if z['x'].shape[1] == len(layout['feature_names']):
                return {'file': Path(source).name, 'frames': len(z['x']), 'dropped': int(z['dropped']), 'ip_axis_corrected': bool(z['ip_axis_corrected'])}
    d, _ = read_case(source, layout)
    tmp = dest.with_suffix('.tmp.npz')
    np.savez(tmp, **d)
    tmp.replace(dest)
    return {'file': Path(source).name, 'frames': len(d['x']), 'dropped': d['dropped'], 'ip_axis_corrected': d['ip_axis_corrected']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit-shots',type=int,default=0)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    started=time.time();out=args.out;out.mkdir(parents=True,exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
    torch.set_num_threads(min(8,args.workers));np.random.seed(args.seed);torch.manual_seed(args.seed)
    files=sorted((args.data/'data/diii_d_train').glob('*.parquet'))
    if not files: raise ValueError('no training shots')
    if args.limit_shots: files=files[:args.limit_shots]
    configure_layout(files,args.workers)
    _,layout=read_case(files[0]);save_json(out/'feature_layout.json',layout)
    save_json(out/'input_audit.json',{'input_columns':INPUTS,'metadata_usage':'source fixed DIII-D; coil/chord names and coil_input_column enforce exact channel identity; numeric geometry enters X','target_columns':LABELS,'excluded_label_columns':FORBIDDEN,'feature_count':len(layout['feature_names']),'thomson':'all core/edge Te/ne channels plus spatial and chord coordinates; finite-sample linear interpolation, zero outside support, observation masks','current':'all 21 actuator/Ip signals, official per-shot Ip timestamp correction; linear interpolation/extrapolation with in-range flags','training_only_fit':True})
    permutation=np.random.RandomState(args.seed).permutation(len(files));nv=max(1,int(.1*len(files)));nt=nv
    splits={'train':[files[i].name for i in permutation[:len(files)-nv-nt]],'val':[files[i].name for i in permutation[len(files)-nv-nt:len(files)-nt]],'test':[files[i].name for i in permutation[len(files)-nt:]]}
    assert not(set(splits['train'])&set(splits['val']) or set(splits['train'])&set(splits['test']) or set(splits['val'])&set(splits['test']))
    save_json(out/'splits.json',splits)
    config=vars(args).copy();config.update(data=str(args.data),out=str(out),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),implementation='DIII-D feature cache and seed-42 shot split (cache stage of the PCA-MLP baseline script)',torch=torch.__version__,numpy=np.__version__)
    save_json(out/'config.json',config)
    checks=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks=((str(f),str(cache/(f.stem+'.npz')),layout) for f in files)
        for i,result in enumerate(pool.map(cache_case,tasks)):
            checks.append(result)
            if (i+1)%100==0 or i+1==len(files):
                status={'stage':'cache','shots':i+1,'total_shots':len(files),'frames':sum(x['frames'] for x in checks),'elapsed_seconds':time.time()-started};save_json(out/'progress.json',status);print(status,flush=True)
    save_json(out/'cache_manifest.json',checks)
    save_json(out/'COMPLETE.json',{'shots':len(files),'features':len(layout['feature_names']),'all_inputs_finite':True,'elapsed_seconds':time.time()-started})


if __name__=='__main__':
    main()
