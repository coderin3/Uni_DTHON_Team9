"""try1 notebook reference model; raw predictions are the default."""
from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from sklearn.pipeline import Pipeline

VERSION = "battery-v9-try1.1"
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
SEED = 42
LABELS = list(range(1, 7))
BASE_SENSORS = ['meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p', 'meas_t1_plus_t2']
GAS_SENSORS = [
    'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcl', 'meas_hcn',
    'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2'
]
META_COLS = ['atmp_t', 'bat_cc', 'bat_class', 'soc_status', 'thermal_t', 'tr_method', 'indu_t', 'indu_t.1']
ALL_RAW_COLS = BASE_SENSORS + GAS_SENSORS + META_COLS

IMAGE_STATS = [
    'img_mean', 'img_std', 'img_max', 'img_p95', 'img_p99',
    'img_hot_05', 'img_hot_07', 'img_hot_085', 'img_hot_095',
    'img_center_mean', 'img_center_max', 'img_center_std',
    'img_center_edge_diff', 'img_top_bottom_diff', 'img_left_right_diff',
    'img_skew_proxy'
]
HOT_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# 1. 이미지 경로 연결
# ---------------------------------------------------------------------------
def validate_frame(frame, labeled=False):
    required = ['sample_id', 'experiment_id', 'relative_time_s', 'image_path'] + BASE_SENSORS[:5]
    if labeled:
        required += ['label']
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    for col in ['sample_id', 'experiment_id']:
        if frame[col].isna().any():
            raise ValueError(f'{col} contains missing values')
    if not frame.sample_id.is_unique:
        raise ValueError('sample_id must be unique within input')
    t = pd.to_numeric(frame.relative_time_s, errors='coerce').to_numpy(float)
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError('relative_time_s must be finite and nonnegative')
    if labeled:
        y = pd.to_numeric(frame.label, errors='raise')
        if not y.isin(LABELS).all():
            raise ValueError('Labels must be integers 1..6')
    if frame.empty:
        raise ValueError('Input is empty')

def read_frame(path, labeled=False):
    if not Path(path).is_file():
        raise FileNotFoundError(f'CSV not found: {Path(path).resolve()}')
    if Path(path).stat().st_size == 0:
        raise ValueError(f'CSV is empty (0 bytes): {path}')
    f = pd.read_csv(path, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    validate_frame(f, labeled)
    f['relative_time_s'] = pd.to_numeric(f.relative_time_s)
    if labeled:
        f['label'] = pd.to_numeric(f.label).astype(int)
    return f.reset_index(drop=True)

def resolve_image_paths(frame, roots):
    roots = list(dict.fromkeys(Path(r).resolve() for r in roots))
    result, unresolved = [], []
    for i, raw in enumerate(frame.image_path):
        if pd.isna(raw):
            raise ValueError('Missing image_path')
        p = Path(str(raw).replace('\\', '/'))
        candidates = [p] if p.is_absolute() else [r / p for r in roots]
        hit = next((q.resolve() for q in candidates if q.is_file()), None)
        result.append(hit)
        if hit is None:
            unresolved.append((i, p.name))
    if unresolved:
        wanted = {name for _, name in unresolved}
        index = {}
        for root in roots:
            for p in root.rglob('*'):
                if p.name in wanted and p.is_file():
                    index.setdefault(p.name, set()).add(p.resolve())
        for i, name in unresolved:
            matches = index.get(name, set())
            if len(matches) != 1:
                raise ValueError(f'Image {name!r}: {len(matches)} matches; fix image_path/root')
            result[i] = next(iter(matches))
    return result

# ---------------------------------------------------------------------------
# 2. 이미지 통계 피처 (샘플마다 독립 계산 → 다른 행 정보 사용 없음)
# ---------------------------------------------------------------------------
def _image_stats(path) -> list[float]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    
    h, w = arr.shape
    if min(h, w) < 4:
        raise ValueError(f"Image is too small: {path}")
    ch1, ch2 = h // 4, 3 * h // 4
    cw1, cw2 = w // 4, 3 * w // 4
    center = arr[ch1:ch2, cw1:cw2]
    
    mask = np.ones((h, w), dtype=bool)
    mask[ch1:ch2, cw1:cw2] = False
    edge_mean = float(arr[mask].mean()) if mask.any() else float(arr.mean())
    
    top_mean = float(arr[:h//2, :].mean())
    bot_mean = float(arr[h//2:, :].mean())
    left_mean = float(arr[:, :w//2].mean())
    right_mean = float(arr[:, w//2:].mean())
    
    mean_val = float(arr.mean())
    std_val = float(arr.std())
    p95_val = float(np.percentile(arr, 95))
    p99_val = float(np.percentile(arr, 99))

    return [
        mean_val,
        std_val,
        float(arr.max()),
        p95_val,
        p99_val,
        float((arr >= 0.5).mean()),
        float((arr >= 0.7).mean()),
        float((arr >= 0.85).mean()),
        float((arr >= 0.95).mean()),
        float(center.mean()),
        float(center.max()),
        float(center.std()),
        float(center.mean()) - edge_mean,
        abs(top_mean - bot_mean),
        abs(left_mean - right_mean),
        (p99_val - mean_val) / (std_val + 1e-5)  # 국소 핫스팟 첨도/왜도 대용 지표
    ]


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = N_JOBS) -> pd.DataFrame:
    """sample_id를 인덱스로 하는 이미지 통계 표를 만듭니다."""
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_image_stats)(p) for p in paths)
    return pd.DataFrame(rows, columns=IMAGE_STATS, index=pd.Index(frame['sample_id'].values, name='sample_id'))


# ---------------------------------------------------------------------------
# 3. 과거 방향 시계열 피처
#    같은 experiment_id 안에서 relative_time_s <= 현재 시점인 관측만 사용합니다.
#    실험 전체 통계(평균·최댓값·길이·순위 등)는 사용하지 않습니다.
# ---------------------------------------------------------------------------
WINDOW_SIGNALS = [
    'meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p',
    'meas_co', 'meas_co2', 'meas_ch4', 'meas_hf', 'meas_hcl', 'meas_hcn',
    'img_max', 'img_p99', 'img_center_max', 'img_hot_085'
]
WINDOWS_S = [15.0, 60.0, 180.0, 600.0]


def _causal_one_experiment_multi(t: np.ndarray, x: np.ndarray, windows_s: list[float]):
    """한 실험(시간순 정렬)에 대해 현재 시점(<= t)까지의 다중 윈도우 통계를 계산합니다."""
    n = len(t)
    results = {}
    end = np.searchsorted(t, t, side='right')  # 현재 시각(<= t)까지만 포함 (규정 준수)

    for w_s in windows_s:
        w_int = int(w_s)
        mean = np.full(n, np.nan)
        vmax = np.full(n, np.nan)
        vmin = np.full(n, np.nan)
        rate = np.full(n, np.nan)

        start = np.searchsorted(t, t - w_s, side='right')
        lag = np.searchsorted(t, t - w_s, side='right') - 1

        for i in range(n):
            seg = x[start[i]:end[i]]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                mean[i] = seg.mean()
                vmax[i] = seg.max()
                vmin[i] = seg.min()
            if lag[i] >= 0:
                dt = t[i] - t[lag[i]]
                if dt > 0:
                    # 시간 간격이 다른 test(5초~15초)에도 동일하게 작동하도록 초당 변화율 계산
                    rate[i] = (x[i] - x[lag[i]]) / dt

        results[f'mean{w_int}s'] = mean
        results[f'max{w_int}s'] = vmax
        results[f'rise{w_int}s'] = x - vmin      # 윈도우 내 최저점 대비 순수 상승량
        results[f'drop{w_int}s'] = vmax - x      # 윈도우 내 최고점 대비 하락량 (열폭주 후 냉각/전압강하 포착)
        results[f'rate{w_int}s'] = rate

    # 최근 15초와 60초의 변화율 차이 (엄밀한 2차 미분은 아님)
    results['accel_15_60'] = results['rate15s'] - results['rate60s']
    return results



def add_all_causal_features(base: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    suffixes = ['accel_15_60']
    for w_s in WINDOWS_S:
        w_int = int(w_s)
        suffixes.extend([f'mean{w_int}s', f'max{w_int}s', f'rise{w_int}s', f'drop{w_int}s', f'rate{w_int}s'])

    for s in WINDOW_SIGNALS:
        for suf in suffixes:
            new_cols[f'{s}_{suf}'] = np.full(len(base), np.nan)

    times = base['relative_time_s'].to_numpy(dtype=float)
    ids = base['sample_id'].astype(str).to_numpy()

    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.lexsort((ids[idx], times[idx]))]
        t = times[idx]

        for s in WINDOW_SIGNALS:
            arr = base[s].to_numpy(dtype=float)[idx]
            res_dict = _causal_one_experiment_multi(t, arr, WINDOWS_S)
            for suffix, vals in res_dict.items():
                new_cols[f'{s}_{suffix}'][idx] = vals

    return pd.DataFrame(new_cols, index=base.index)


# ---------------------------------------------------------------------------
# 4. 전체 피처 조립
# ---------------------------------------------------------------------------
def numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame) -> pd.DataFrame:
    validate_frame(frame)
    frame = frame.reset_index(drop=True).copy()
    # Keep a fixed schema if an optional raw column is absent.
    for col in ALL_RAW_COLS:
        if col not in frame:
            frame[col] = np.nan
    base = frame[['sample_id', 'experiment_id']].copy()
    base['relative_time_s'] = pd.to_numeric(frame['relative_time_s'], errors='coerce')

    avail_raw = [c for c in ALL_RAW_COLS if c in frame.columns]
    base[avail_raw] = numeric(frame, avail_raw)

    # 1. 물리·화학적 도메인 파생 변수
    base['t_diff_1_2'] = base['meas_t1'] - base['meas_t2']
    base['t_max_1_2'] = np.fmax(base['meas_t1'], base['meas_t2'])
    base['t_min_1_2'] = np.fmin(base['meas_t1'], base['meas_t2'])
    base['t_rise_ambient'] = base['t_max_1_2'] - base['atmp_t']
    base['power_va'] = base['meas_v'] * base['meas_a']
    
    # 노트북의 가스 조합 피처
    base['total_gas'] = base[GAS_SENSORS].sum(axis=1)
    base['toxic_acid_gas'] = base['meas_hf'] + base['meas_hcl'] + base['meas_hcn']
    base['co_co2_ratio'] = base['meas_co'] / (base['meas_co2'] + 1e-5)
    base['ch4_co2_ratio'] = base['meas_ch4'] / (base['meas_co2'] + 1e-5)
    base['hf_ratio'] = base['meas_hf'] / (base['total_gas'] + 1e-5)
    
    # 전압 상태 피처
    base['is_volt_drop'] = (base['meas_v'] < 3.0).astype(float)
    base['is_volt_collapsed'] = (base['meas_v'] < 1.0).astype(float)
    base['indu_t_notna'] = base['indu_t'].notna().astype(float)

    # 2. 이미지 피처 결합
    img = image_features.reindex(frame['sample_id'].values)
    assert img.notna().all(axis=1).all(), '이미지 피처가 없는 샘플이 있습니다.'
    base[IMAGE_STATS] = img.values

    # 3. 멀티모달 교호작용 피처 (온도 x 가스 x 열화상)
    base['img_hotspot_contrast'] = base['img_max'] - base['img_mean']
    base['img_center_hotspot'] = base['img_center_max'] - base['img_center_mean']
    base['temp_x_gas'] = base['t_rise_ambient'] * np.log1p(np.maximum(base['total_gas'], 0))
    base['temp_x_img'] = base['t_max_1_2'] * base['img_p99']

    # 4. 인과적 시계열 피처 생성
    causal = add_all_causal_features(base)

    # 5. 시계열 가속도 x 가스 결합 (열폭주 폭발 순간 포착)
    causal['t1_rate15_x_hf'] = causal['meas_t1_rate15s'] * np.log1p(np.maximum(base['meas_hf'], 0))

    # 1) 온도 급상승 × 전압 급강하 결합 (열폭주 직전 순간 포착)
    causal['t1_rate_x_v_drop'] = causal['meas_t1_rate15s'] * (-causal['meas_v_rate15s'])
    
    # 2) 노트북의 관측값 누적합. 시간 적분이나 물리적인 누적 발생량은 아님.
    # 시간순 정렬 후 누적합 계산
    # Notebook-compatible row sums for unique timestamps. Equal-time batches
    # include all simultaneous observations; tie order is deterministic.
    for column, target in [('total_gas', 'cum_total_gas'), ('meas_hf', 'cum_hf')]:
        causal[target] = causal_sum(base, column)

    feature_cols = [c for c in base.columns if c not in ('sample_id', 'experiment_id')]
    return pd.concat([base[feature_cols], causal], axis=1).replace([np.inf, -np.inf], np.nan)


def causal_sum(base, column):
    result = np.full(len(base), np.nan)
    times = base.relative_time_s.to_numpy(float)
    ids = base.sample_id.astype(str).to_numpy()
    ordered = base.sort_values(['experiment_id', 'relative_time_s', 'sample_id'], kind='mergesort')
    # Preserve pandas groupby.cumsum's numerical accumulation, as in try1.
    sums = ordered.groupby('experiment_id')[column].cumsum()
    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.lexsort((ids[idx], times[idx]))]
        t = times[idx]
        summed = sums.reindex(base.index[idx]).to_numpy()
        end = np.searchsorted(t, t, side='right')
        result[idx] = summed[end - 1]
    return result


def build_model(class_weight='balanced', seed=SEED, n_jobs=N_JOBS, **overrides):
    from lightgbm import LGBMClassifier
    # Preserve the actual notebook settings. subsample_freq=0 means the
    # notebook's subsample=.8 did not activate row bagging; do not silently change it.
    params = dict(n_estimators=600, learning_rate=.03, num_leaves=31,
                  subsample_for_bin=50000, colsample_bytree=.8,
                  subsample=.8, subsample_freq=0, min_child_samples=30,
                  reg_alpha=0., reg_lambda=0., class_weight=class_weight,
                  random_state=seed, n_jobs=n_jobs, verbose=-1)
    params.update(overrides)
    return Pipeline([('classifier', LGBMClassifier(**params))])


def predict_proba6(pipeline, x):
    p = np.zeros((len(x), 6), dtype=float)
    p[:, np.asarray(pipeline.classes_, dtype=int)-1] = pipeline.predict_proba(x)
    return p


def ensemble_proba(models, x):
    return np.mean([predict_proba6(m, x) for m in models], axis=0)


def predict_labels(bundle, x, raw=False):
    if bundle.get('version') != VERSION:
        raise ValueError('Incompatible model. Use the v9 model.py and retrain.')
    missing = set(bundle['features']) - set(x.columns)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    p = ensemble_proba(bundle['models'], x[bundle['features']])
    scale = np.ones(6) if raw else np.asarray(bundle['class_scale'], dtype=float)
    if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Class scales must be six finite positive numbers')
    return (p * scale).argmax(axis=1)+1
