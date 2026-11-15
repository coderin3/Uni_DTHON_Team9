"""Generate submission using the same try1 feature and probability code."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import model as M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('weights/v9_try1.joblib'))
    parser.add_argument('--output', type=Path, default=Path('work/submission_v9_try1.csv'))
    parser.add_argument('--sample_submission', type=Path)
    parser.add_argument('--reference_submission', type=Path, help='Optional comparison to your old notebook CSV; no fitting')
    parser.add_argument('--raw', action='store_true', help='Ignore saved scales; same as default for uncalibrated model')
    parser.add_argument('--n_jobs', type=int, default=M.N_JOBS)
    args = parser.parse_args()
    if args.reference_submission and args.reference_submission.resolve() == args.output.resolve():
        parser.error('Use different paths for --output and --reference_submission; preserve your original CSV')
    frame = M.read_frame(args.input)
    bundle = joblib.load(args.model)
    if bundle.get('version') != M.VERSION:
        raise ValueError('Wrong model version: v7/v8 weights cannot be used with v9')
    paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
    img = M.extract_image_features(frame, paths, args.n_jobs)
    x = M.build_features(frame, img)
    labels = M.predict_labels(bundle, x, raw=args.raw)
    pred = pd.Series(labels, index=frame.sample_id.astype(str).to_numpy())
    ids = frame.sample_id.astype(str).tolist()
    if args.sample_submission:
        template = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})
        if 'sample_id' not in template or template.sample_id.isna().any() or not template.sample_id.is_unique:
            raise ValueError('Invalid sample submission IDs')
        ids = template.sample_id.astype(str).tolist()
        if set(ids) != set(pred.index):
            raise ValueError('Input and sample submission IDs differ')
    result = pd.DataFrame({'sample_id': ids, 'label': pred.loc[ids].to_numpy(int)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Saved: {args.output.resolve()} ({len(result)} rows)')
    print('Predicted label counts:', result.label.value_counts().sort_index().to_dict())
    if args.reference_submission:
        ref = pd.read_csv(args.reference_submission, dtype={'sample_id': 'string'})
        if not {'sample_id', 'label'} <= set(ref) or not ref.sample_id.is_unique or set(ref.sample_id) != set(ids):
            raise ValueError('Reference submission must have the same unique IDs')
        old = pd.to_numeric(ref.set_index('sample_id').loc[ids, 'label'], errors='raise').to_numpy()
        if not np.isin(old, M.LABELS).all():
            raise ValueError('Reference submission contains invalid labels')
        changed = int((old != result.label.to_numpy()).sum())
        print(f'Reference comparison: {changed}/{len(result)} changed '
              f'({changed/len(result):.4%}); this is prediction agreement, not accuracy.')


if __name__ == '__main__':
    main()
