"""Check a trained model's causality, row-order invariance and image sensitivity."""
import argparse
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import model as M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--n_jobs', type=int, default=M.N_JOBS)
    parser.add_argument('--require_image_effect', action='store_true')
    args = parser.parse_args()
    frame = M.read_frame(args.input)
    bundle = joblib.load(args.model)
    paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
    img = M.extract_image_features(frame, paths, args.n_jobs)

    def features(f):
        x = M.build_features(f, img)
        x.index = f.sample_id.to_numpy()
        return x

    full = features(frame)
    predictions = pd.Series(M.predict_labels(bundle, full), index=frame.sample_id.to_numpy())

    def check(name, f):
        x = features(f)
        pd.testing.assert_frame_equal(full.loc[x.index], x, check_exact=False, rtol=1e-12, atol=1e-12)
        p = M.predict_labels(bundle, x)
        np.testing.assert_array_equal(predictions.loc[x.index].to_numpy(), p)
        print(f'PASS {name}: {len(f)} rows')

    check('shuffled order', frame.sample(frac=1, random_state=42).reset_index(drop=True))
    for fraction in (.25, .5, .75):
        cut = frame.groupby('experiment_id').relative_time_s.transform(lambda s: s.quantile(fraction))
        check(f'future removed at fraction {fraction}', frame[frame.relative_time_s <= cut].reset_index(drop=True))
    for _, group in frame.groupby('experiment_id', sort=False):
        check('single experiment', group.reset_index(drop=True))
    # Independent per-image caches are reused above; all temporal features are rebuilt.
    with tempfile.TemporaryDirectory() as tmp:
        blank = Path(tmp)/'blank.png'
        with Image.open(paths[0]) as original:
            Image.new(original.mode, original.size).save(blank)
        stats = M._image_stats(blank)
        blank_img = pd.DataFrame(np.tile(stats, (len(frame), 1)), columns=M.IMAGE_STATS, index=frame.sample_id)
        blank_pred = M.predict_labels(bundle, M.build_features(frame, blank_img))
        changes = int((blank_pred != predictions.to_numpy()).sum())
        print(f'Blank-image prediction changes: {changes}/{len(frame)} ({changes/len(frame):.4%})')
        if args.require_image_effect and changes == 0:
            raise AssertionError('No prediction changed under blank-image replacement')
    print('All requested checks passed. Arbitrary removal of past observations is not an invariant.')


if __name__ == '__main__':
    main()
