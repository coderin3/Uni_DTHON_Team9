"""Reproduce try1's raw balanced model; additional experiments are opt-in."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
import model as M


def metrics(y, pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=M.LABELS, zero_division=0)
    return dict(macro_f1=float(f1.mean()), precision=precision.tolist(),
                recall=recall.tolist(), f1=f1.tolist(), support=support.tolist(),
                confusion_matrix=confusion_matrix(y, pred, labels=M.LABELS).tolist())


def show(name, y, pred):
    result = metrics(y, pred)
    print(f'{name}: Macro F1={result["macro_f1"]:.6f} | '+
          ' '.join(f'{i+1}:{f:.3f}' for i, f in enumerate(result['f1'])), flush=True)
    return result


def load_split(args, split):
    f = M.read_frame(args.data_dir/split/f'{split}.csv', labeled=True)
    paths = M.resolve_image_paths(f, [args.data_dir/split, args.data_dir])
    digest = hashlib.sha256(inspect.getsource(M._image_stats).encode())
    for sid, path in zip(f.sample_id, paths):
        stat = path.stat()
        digest.update(json.dumps([str(sid), str(path), stat.st_size, stat.st_mtime_ns]).encode())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir/f'{split}-{digest.hexdigest()[:20]}.joblib'
    img = None
    if cache.exists() and not args.refresh_cache:
        candidate = joblib.load(cache)
        if (isinstance(candidate, pd.DataFrame) and candidate.index.is_unique
                and candidate.columns.tolist() == M.IMAGE_STATS
                and set(candidate.index) == set(f.sample_id)
                and np.isfinite(candidate.to_numpy()).all()):
            img = candidate
    if img is None:
        print(f'[{split}] Extracting notebook image features', flush=True)
        img = M.extract_image_features(f, paths, args.n_jobs)
        joblib.dump(img, cache)
    x = M.build_features(f, img)
    print(f'[{split}] {len(f)} rows / {f.experiment_id.nunique()} experiments / {x.shape[1]} features', flush=True)
    return f, x, f.label.to_numpy(int)


def fit_models(x, y, args):
    if len(np.unique(y)) < 2:
        raise ValueError('Training fold has fewer than two classes')
    cw = None if args.class_weight == 'none' else 'balanced'
    result = []
    for seed in args.seeds:
        model = M.build_model(cw, seed, args.n_jobs, n_estimators=args.n_estimators)
        model.fit(x, y)
        result.append(model)
    return result


def oof_predictions(f, x, y, args):
    n = min(args.folds, f.experiment_id.nunique())
    if n < 2:
        raise ValueError('At least two experiments are needed for OOF')
    p = np.zeros((len(f), 6))
    records = []
    cv = GroupKFold(n_splits=n)
    for fold, (a, b) in enumerate(cv.split(x, y, f.experiment_id), 1):
        print(f'OOF {fold}/{n}', flush=True)
        models = fit_models(x.iloc[a], y[a], args)
        p[b] = M.ensemble_proba(models, x.iloc[b])
        records.append(show(f'OOF fold {fold}', y[b], p[b].argmax(1)+1))
    return p, records


def tune_scales(y, p):
    """Optional bounded positive multiplier search; fitted-score is not holdout."""
    from scipy.optimize import minimize
    # Match the notebook's bounded candidate, but reject non-improvements.
    def loss(scale):
        return -metrics(y, (p*scale).argmax(1)+1)['macro_f1']
    weights = np.ones(6)
    baseline = loss(weights)
    result = minimize(loss, weights, method='Nelder-Mead',
                      bounds=[(.03, 2.)]*6, options={'maxiter': 800})
    if (np.isfinite(result.x).all() and (result.x > 0).all()
            and loss(result.x) < baseline-1e-12):
        weights = result.x
    return weights.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--model_out', type=Path, default=Path('weights/v9_try1.joblib'))
    parser.add_argument('--report_dir', type=Path, default=Path('reports/v9_try1'))
    parser.add_argument('--cache_dir', type=Path, default=Path('.cache/try1_v9'))
    parser.add_argument('--refresh_cache', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=M.N_JOBS)
    parser.add_argument('--class_weight', choices=['none', 'balanced'], default='balanced')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--n_estimators', type=int, default=600)
    parser.add_argument('--drop', nargs='*', default=[])
    parser.add_argument('--cv', action='store_true', help='Optional Train-only GroupKFold evaluation')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--calibration', choices=['none', 'validation', 'oof'], default='none')
    parser.add_argument('--refit_train_val', action='store_true')
    parser.add_argument('--eval', action='store_true', help='Reports only, no model save')
    args = parser.parse_args()
    if args.folds < 2 or args.n_estimators < 1:
        parser.error('folds >= 2 and n_estimators >= 1 are required')
    started = time.time()
    train, xt, yt = load_split(args, 'train')
    val, xv, yv = load_split(args, 'validation')
    if set(train.experiment_id) & set(val.experiment_id):
        raise ValueError('Train and Validation experiments overlap')
    if set(train.sample_id) & set(val.sample_id):
        raise ValueError('Train and Validation sample IDs overlap')
    if set(yt) != set(M.LABELS):
        raise ValueError('Training split must contain all six labels')
    features = [c for c in xt if not any(c == d or c.startswith(d+'_') for d in args.drop)]
    if not features:
        raise ValueError('All features were dropped')
    xt, xv = xt[features], xv[features]
    print(f'Configuration: {len(features)} features, class_weight={args.class_weight}, '
          f'seeds={args.seeds}, augmentation=none, calibration={args.calibration}', flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = dict(version=M.VERSION, config=config, features=features)
    p_oof = None
    if args.cv or args.calibration == 'oof':
        p_oof, folds = oof_predictions(train, xt, yt, args)
        report['oof_folds'] = folds
        report['oof_raw'] = show('Train pooled OOF raw', yt, p_oof.argmax(1)+1)
        report['oof_fold_mean'] = float(np.mean([f['macro_f1'] for f in folds]))
    print('Fitting Train-only model...', flush=True)
    models = fit_models(xt, yt, args)
    p_val = M.ensemble_proba(models, xv)
    report['validation_raw'] = show('Validation raw', yv, p_val.argmax(1)+1)
    scale = [1.]*6
    if args.calibration == 'validation':
        scale = tune_scales(yv, p_val)
    elif args.calibration == 'oof':
        scale = tune_scales(yt, p_oof)
        report['oof_calibrated_selection_score'] = show(
            'OOF after calibration (selection score)', yt, (p_oof*scale).argmax(1)+1)
    name = 'Validation calibrated (selection score)' if args.calibration == 'validation' else 'Validation selected'
    report['validation_selected'] = show(name, yv, (p_val*scale).argmax(1)+1)
    report['validation_used_for_calibration'] = args.calibration == 'validation'
    report['class_scale'] = scale
    print('Class scales:', scale, flush=True)
    # Reports always describe the Train-only model, even if final refit is requested.
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for variant in ['raw', 'selected']:
        m = report[f'validation_{variant}']
        pd.DataFrame(m['confusion_matrix'], index=M.LABELS, columns=M.LABELS).to_csv(
            args.report_dir/f'validation_confusion_{variant}.csv')
    summary = []
    for key, value in report.items():
        if isinstance(value, dict) and 'macro_f1' in value:
            summary.append(dict(evaluation=key, macro_f1=value['macro_f1'],
                                **{f'f1_{i+1}': v for i, v in enumerate(value['f1'])}))
    pd.DataFrame(summary).to_csv(args.report_dir/'summary.csv', index=False)
    importance = np.mean([m.named_steps['classifier'].booster_.feature_importance('gain') for m in models], axis=0)
    pd.DataFrame(dict(feature=features, gain=importance)).sort_values('gain', ascending=False).to_csv(
        args.report_dir/'feature_importance.csv', index=False)
    import lightgbm, sklearn
    environment = dict(python=platform.python_version(), numpy=np.__version__,
                       pandas=pd.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__)
    report.update(environment=environment, elapsed_seconds=time.time()-started)
    (args.report_dir/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Reports: {args.report_dir.resolve()}', flush=True)
    if args.eval:
        return
    if args.refit_train_val:
        print('Refitting on Train+Validation (not notebook reproduction mode)', flush=True)
        models = fit_models(pd.concat([xt, xv], ignore_index=True), np.r_[yt, yv], args)
    bundle = dict(version=M.VERSION, models=models, features=features, class_scale=scale,
                  calibration=args.calibration, refit_train_val=args.refit_train_val,
                  config=config, environment=environment)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    print(f'Saved: {args.model_out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
