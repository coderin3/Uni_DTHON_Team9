"""Analyze labeled Validation only. Never fits or changes a model.

Copy beside v9 model.py. Reports contain aggregate counts and local experiment
aliases, not original sample IDs. Labels are used for diagnostics only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import model as M

BUCKETS = ['0-30s', '30-60s', '60-120s', '120-300s', '300s+', 'start_unknown']
TARGETS = [(3, 2), (5, 4)]


def transition_context(frame):
    """Age since the FIRST OBSERVED sample of a new labeled stage segment.

    This is not the exact physical transition time. The first observed stage
    in an experiment has unknown onset. Equal-time labels must agree.
    """
    f = frame.drop(columns=['_age', '_bucket', '_position'], errors='ignore').reset_index(drop=True).copy()
    if (f.groupby(['experiment_id', 'relative_time_s']).label.nunique() > 1).any():
        raise ValueError('Conflicting labels at the same experiment/time; transition-age analysis is ambiguous')
    unique = (f.sort_values(['experiment_id', 'relative_time_s', 'sample_id'])
              .drop_duplicates(['experiment_id', 'relative_time_s']).copy())
    previous = unique.groupby('experiment_id').label.shift()
    changed = unique.label.ne(previous)
    unique['_segment'] = changed.groupby(unique.experiment_id).cumsum()
    groups = unique.groupby(['experiment_id', '_segment'], sort=False)
    first_time = groups.relative_time_s.transform('min')
    observed = previous.notna() & changed
    unique['_known'] = observed.groupby([unique.experiment_id, unique['_segment']]).transform('max')
    unique['_age'] = np.where(unique['_known'], unique.relative_time_s-first_time, np.nan)
    unique['_bucket'] = pd.cut(unique['_age'], [0, 30, 60, 120, 300, np.inf],
                               right=False, labels=BUCKETS[:-1]).astype(object)
    unique['_bucket'] = unique['_bucket'].fillna('start_unknown')
    f['_position'] = np.arange(len(f))
    f = f.merge(unique[['experiment_id', 'relative_time_s', '_age', '_bucket']],
                on=['experiment_id', 'relative_time_s'], how='left', validate='many_to_one')
    return f.sort_values('_position').reset_index(drop=True)


def analyze(frame, probability, name):
    y = frame.label.to_numpy(int)
    pred = probability.argmax(axis=1)+1
    summary = {'model': name, 'macro_f1': float(f1_score(y, pred, labels=M.LABELS,
                average='macro', zero_division=0)), 'all_errors': int((y != pred).sum())}
    by_exp, by_age, by_probability = [], [], []
    for actual, predicted in TARGETS:
        target = f'{actual}_to_{predicted}'
        positive = y == actual
        error = positive & (pred == predicted)
        summary[f'{target}_errors'] = int(error.sum())
        for alias, positions in frame.groupby('_experiment_alias', sort=True).indices.items():
            n = int(positive[positions].sum())
            if n:
                count = int(error[positions].sum())
                by_exp.append(dict(model=name, target=target, experiment=alias,
                                   actual_class_rows=n, target_errors=count, error_rate=count/n))
        ages = frame['_bucket'].to_numpy()
        for bucket in BUCKETS:
            selected = positive & (ages == bucket)
            count = int((error & selected).sum())
            n = int(selected.sum())
            by_age.append(dict(model=name, target=target, bucket=bucket, actual_class_rows=n,
                               target_errors=count, error_rate=count/n if n else np.nan,
                               share_of_target_errors=count/error.sum() if error.any() else np.nan))
        # Ratios are diagnostics of decision margins, not recommended multipliers.
        if error.any():
            correct_probability = probability[error, actual-1]
            wrong_probability = probability[error, predicted-1]
            ratio = wrong_probability/np.maximum(correct_probability, 1e-15)
            second = np.argsort(probability[error], axis=1)[:, -2]+1
            by_probability.append(dict(model=name, target=target, rows=int(error.sum()),
                                       median_actual_probability=float(np.median(correct_probability)),
                                       median_predicted_probability=float(np.median(wrong_probability)),
                                       median_wrong_to_actual_ratio=float(np.median(ratio)),
                                       actual_is_second_choice=float((second == actual).mean())))
        counts = sorted([r['target_errors'] for r in by_exp if r['target'] == target], reverse=True)
        summary[f'{target}_top3_experiment_share'] = sum(counts[:3])/error.sum() if error.any() else 0.
    return summary, by_exp, by_age, by_probability, pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('reports/error_diagnostics'))
    parser.add_argument('--n_jobs', type=int, default=4)
    args = parser.parse_args()
    names = [path.stem for path in args.models]
    if len(names) != len(set(names)):
        parser.error('Model filenames must have different stems')
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new --output_dir')
    f = M.read_frame(args.data_dir/'validation/validation.csv', labeled=True)
    f = transition_context(f)
    aliases = {exp: f'EXP_{i+1:03d}' for i, exp in enumerate(sorted(f.experiment_id.unique()))}
    f['_experiment_alias'] = f.experiment_id.map(aliases)
    print('Extracting Validation image/features once. No training or Test labels are used.', flush=True)
    paths = M.resolve_image_paths(f, [args.data_dir/'validation', args.data_dir])
    img = M.extract_image_features(f, paths, args.n_jobs)
    x = M.build_features(f, img)
    summaries, experiments, buckets, margins, predictions = [], [], [], [], {}
    for path in args.models:
        bundle = joblib.load(path)
        if bundle.get('version') != M.VERSION:
            raise ValueError(f'{path}: model/code version mismatch')
        scale = np.asarray(bundle['class_scale'], dtype=float)
        if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid class scales')
        p = M.ensemble_proba(bundle['models'], x[bundle['features']])*scale
        p /= p.sum(axis=1, keepdims=True)
        summary, exp, age, margin, pred = analyze(f, p, path.stem)
        summaries.append(summary); experiments.extend(exp); buckets.extend(age); margins.extend(margin)
        predictions[path.stem] = pred
        print(f'\n{path.stem}: Macro F1={summary["macro_f1"]:.6f}', flush=True)
        for actual, wrong in TARGETS:
            target = f'{actual}_to_{wrong}'
            rows = pd.DataFrame([r for r in exp if r['target'] == target])
            print(f'\n{target}: {summary[target+"_errors"]} errors; '
                  f'top 3 experiment share={summary[target+"_top3_experiment_share"]:.1%}')
            if not rows.empty:
                print(rows.sort_values('target_errors', ascending=False).head(5).to_string(index=False))
            print(pd.DataFrame([r for r in age if r['target'] == target]).to_string(index=False))
    changes = []
    first = names[0]
    y = f.label.to_numpy(int)
    for name in names[1:]:
        a, b = predictions[first], predictions[name]
        changed = a != b
        for actual in M.LABELS:
            indices = np.flatnonzero(changed & (y == actual))
            pairs = pd.DataFrame({'from_prediction': a[indices], 'to_prediction': b[indices]})
            for (old, new), count in pairs.value_counts(sort=False).items():
                changes.append(dict(reference_model=first, comparison_model=name,
                                    actual=int(actual), from_prediction=int(old),
                                    to_prediction=int(new), rows=int(count)))
    args.output_dir.mkdir(parents=True)
    for name, records in [('summary', summaries), ('experiment_errors', experiments),
                          ('transition_buckets', buckets), ('probability_margins', margins)]:
        pd.DataFrame(records).to_csv(args.output_dir/f'{name}.csv', index=False)
    pd.DataFrame(changes, columns=['reference_model','comparison_model','actual','from_prediction',
                                   'to_prediction','rows']).to_csv(args.output_dir/'prediction_changes.csv', index=False)
    (args.output_dir/'NOTES.txt').write_text(
        'Labels are used for diagnostics only and are not model inputs.\n'
        'Transition age starts at the first observed sample with a new label, not the exact physical transition.\n'
        'The first stage segment in each experiment has unknown onset.\n'
        'Rate denominators are the actual-class rows in each bucket/experiment.\n'
        'No sample IDs, original experiment IDs, raw sensor values or images are exported.\n'
        'Experiment aliases are local to this Validation input, not anonymization guarantees.\n'
        'Probability margins do not establish that a multiplier will generalize.\n', encoding='utf-8')
    print(f'\nSaved aggregate reports: {args.output_dir.resolve()}', flush=True)
    print('Model files and existing submission CSVs were not modified.', flush=True)


if __name__ == '__main__':
    main()
