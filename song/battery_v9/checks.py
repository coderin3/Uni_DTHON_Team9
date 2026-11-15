"""Synthetic tests; --smoke also runs real LightGBM train/save/predict."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import model as M


def synthetic(experiments=3, offset=0, duplicates=False):
    rng = np.random.default_rng(23)
    rows = []
    for e in range(offset, offset+experiments):
        for j in range(18):
            for copy in range(2 if duplicates and j in (0, 4) else 1):
                row = dict(sample_id=f'{e:03d}-{j:03d}-{copy}', experiment_id=f'e{e}',
                           relative_time_s=float(j*5), image_path='shared.png', label=j//3+1)
                row.update({s: float(j+rng.uniform(.1, 2.)) for s in M.ALL_RAW_COLS})
                row['meas_a'] = 0.
                if j < 3:
                    row['indu_t'] = np.nan
                row['indu_t.1'] = np.nan
                for k, c in enumerate(['bat_class', 'bat_cc', 'soc_status', 'tr_method']):
                    row[c] = (e+k)%3
                rows.append(row)
    return pd.DataFrame(rows)


def fake_images(f):
    values = np.random.default_rng(8).random((len(f), len(M.IMAGE_STATS)))
    return pd.DataFrame(values, columns=M.IMAGE_STATS, index=f.sample_id)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.f = synthetic(duplicates=True)
        self.img = fake_images(self.f)

    def features(self, f):
        x = M.build_features(f, self.img)
        x.index = f.sample_id.to_numpy()
        return x.sort_index()

    def test_feature_schema_and_no_label_input(self):
        x = self.features(self.f)
        self.assertEqual(x.shape[1], 377)
        self.assertTrue(x.columns.is_unique)
        self.assertFalse({'label', 'sample_id', 'experiment_id'} & set(x.columns))
        f = self.f.copy(); f['label'] = 7-f.label
        pd.testing.assert_frame_equal(x, self.features(f))

    def test_shuffle(self):
        pd.testing.assert_frame_equal(self.features(self.f), self.features(self.f.sample(frac=1, random_state=6)))

    def test_future_removal(self):
        full = self.features(self.f)
        for cutoff in (0, 15, 30, 60):
            f = self.f[self.f.relative_time_s <= cutoff]
            part = self.features(f)
            pd.testing.assert_frame_equal(full.loc[part.index], part)

    def test_experiment_isolation(self):
        full = self.features(self.f)
        for _, f in self.f.groupby('experiment_id'):
            part = self.features(f)
            pd.testing.assert_frame_equal(full.loc[part.index], part)

    def test_missing_optional_schema(self):
        f = self.f.drop(columns=['indu_t.1', 'bat_cc'])
        x = self.features(f)
        self.assertEqual(x.shape[1], 377)
        self.assertTrue(x['indu_t.1'].isna().all())

    def test_causal_window_values(self):
        t = np.array([0., 5., 15., 30.])
        x = np.array([3., 2., 1., 5.])
        a = M._causal_one_experiment_multi(t, x, M.WINDOWS_S)
        self.assertEqual(a['mean15s'][2], 1.5)
        self.assertEqual(a['rise15s'][2], 0.)
        self.assertEqual(a['drop15s'][2], 1.)
        self.assertAlmostEqual(a['rate15s'][2], -2/15)

    def test_image_statistics(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'image.png'
            Image.fromarray(np.full((12, 16, 3), 128, dtype=np.uint8)).save(path)
            values = M._image_stats(path)
            self.assertEqual(len(values), 16)
            self.assertTrue(np.isfinite(values).all())


def smoke():
    import joblib
    with tempfile.TemporaryDirectory(prefix='battery-v9-') as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(5)
        for split, count, offset in [('train', 6, 0), ('validation', 2, 20), ('test', 2, 40)]:
            folder = root/split; folder.mkdir()
            Image.fromarray(rng.integers(0, 256, (16, 20, 3), dtype=np.uint8)).save(folder/'shared.png')
            f = synthetic(count, offset, duplicates=True)
            if split == 'test':
                f = f.drop(columns='label')
            f.to_csv(folder/f'{split}.csv', index=False)
        code = Path(__file__).resolve().parent
        base = [sys.executable, str(code/'train.py'), '--data_dir', str(root),
                '--n_estimators', '8', '--n_jobs', '1', '--cache_dir', str(root/'cache'),
                '--report_dir', str(root/'reports'), '--model_out', str(root/'model.joblib')]
        subprocess.run(base, check=True)
        bundle = joblib.load(root/'model.joblib')
        assert bundle['class_scale'] == [1.]*6 and len(bundle['features']) == 377
        assert not bundle['refit_train_val']
        test = M.read_frame(root/'test/test.csv')
        template = test[['sample_id']].iloc[::-1].copy(); template['label'] = 0
        template.to_csv(root/'template.csv', index=False)
        cmd = [sys.executable, str(code/'test.py'), '--input', str(root/'test/test.csv'),
               '--image_root', str(root/'test'), '--model', str(root/'model.joblib'),
               '--sample_submission', str(root/'template.csv'), '--n_jobs', '1',
               '--output', str(root/'submission.csv')]
        subprocess.run(cmd, check=True)
        pred = pd.read_csv(root/'submission.csv')
        assert pred.sample_id.tolist() == template.sample_id.tolist()
        assert pred.label.isin(M.LABELS).all()
        # Compare against a saved reference prediction file.
        subprocess.run(cmd+['--output', str(root/'submission_second.csv'),
                            '--reference_submission', str(root/'submission.csv')], check=True)
        subprocess.run([sys.executable, str(code/'check_model.py'), '--input', str(root/'test/test.csv'),
                        '--image_root', str(root/'test'), '--model', str(root/'model.joblib'),
                        '--n_jobs', '1'], check=True)
        # Exercise opt-in paths; this is not the recommended first training command.
        subprocess.run(base+['--cv', '--folds', '3', '--calibration', 'oof',
                             '--refit_train_val', '--seeds', '42', '7'], check=True)
        subprocess.run(base+['--calibration', 'validation', '--eval', '--drop', 'indu_t.1'], check=True)
        print('PASS: real LightGBM training, calibration modes, CV, refit, serialization and submission.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FeatureTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.smoke:
        smoke()
