"""Package existing v9 code and weights; does not retrain or upload files."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shlex
import shutil
import tempfile
from urllib.parse import urlparse
import zipfile
import joblib
import numpy as np
import pandas as pd
import model as M

LIMIT_BYTES = 25_000_000
PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'Pillow', 'joblib', 'scipy']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def os_description():
    release = Path('/etc/os-release')
    if release.exists():
        for line in release.read_text().splitlines():
            if line.startswith('PRETTY_NAME='):
                return line.split('=', 1)[1].strip('"')
    return platform.platform()


def verify(bundle, data_dir, reference, n_jobs):
    f = M.read_frame(data_dir/'test/test.csv')
    paths = M.resolve_image_paths(f, [data_dir/'test', data_dir])
    print('Checking predictions against the existing submission...', flush=True)
    img = M.extract_image_features(f, paths, n_jobs)
    pred = pd.Series(M.predict_labels(bundle, M.build_features(f, img)), index=f.sample_id)
    ref = pd.read_csv(reference, dtype={'sample_id': 'string'})
    if not {'sample_id', 'label'} <= set(ref) or not ref.sample_id.is_unique or set(ref.sample_id) != set(pred.index):
        raise ValueError('Reference CSV IDs must match the Test CSV exactly')
    expected = pd.to_numeric(ref.label, errors='raise').to_numpy()
    if not np.isin(expected, M.LABELS).all():
        raise ValueError('Reference CSV has invalid labels')
    changed = int((pred.loc[ref.sample_id].to_numpy() != expected).sum())
    if changed:
        raise ValueError(f'{changed}/{len(ref)} predictions differ. No ZIP created; check code and weights.')
    print(f'PASS: all {len(ref)} predictions match.', flush=True)
    return f'기존 제출 CSV와 Test {len(ref):,}행의 예측을 비교하여 전부 일치함을 확인했습니다. 원본 CSV SHA-256: `{sha256(reference)}`.'


def readme(bundle, env, model_hash, model_size, weights_url, verification, source_hashes):
    cfg = bundle['config']
    classifiers = [p.named_steps['classifier'] for p in bundle['models']]
    params = classifiers[0].get_params()
    seeds = [p.get_params()['random_state'] for p in classifiers]
    train_args = ['python', 'train.py', '--data_dir', './data/battery_data_light',
                  '--model_out', './weights/model.joblib', '--report_dir', './reports',
                  '--class_weight', cfg.get('class_weight', 'balanced'),
                  '--seeds'] + [str(s) for s in seeds]
    train_args += ['--n_estimators', str(params['n_estimators']), '--n_jobs', '4',
                   '--calibration', bundle.get('calibration', 'none')]
    if bundle.get('refit_train_val'):
        train_args += ['--refit_train_val']
    if cfg.get('drop'):
        train_args += ['--drop'] + cfg['drop']
    if cfg.get('cv'):
        train_args += ['--cv']
    if cfg.get('cv') or bundle.get('calibration') == 'oof':
        train_args += ['--folds', str(cfg.get('folds', 5))]
    train_command = shlex.join(train_args)
    calibration = bundle.get('calibration', 'none')
    fit_data = 'Train+Validation 합산' if bundle.get('refit_train_val') else 'Train만'
    if calibration == 'validation':
        calibration_note = 'Validation으로 클래스 확률 배율을 선택했습니다. 보정 후 Validation 점수는 독립 평가가 아니라 선택 점수입니다.'
    elif calibration == 'oof':
        calibration_note = 'Train의 실험 단위 GroupKFold OOF로 클래스 확률 배율을 선택했습니다. 보정 후 OOF 점수는 선택 점수입니다.'
    else:
        calibration_note = '확률 배율 최적화와 시계열 후처리는 사용하지 않았습니다.'
    if model_size > LIMIT_BYTES:
        weight_location = f'가중치가 25MB를 초과하여 ZIP에 포함하지 않았습니다. [가중치 다운로드]({weights_url})에서 받아 **`weights/model.joblib`**로 저장한 뒤 추론하세요.'
    else:
        weight_location = '가중치는 ZIP의 `weights/model.joblib`에 포함되어 있습니다.'
    package_table = '\n'.join(f'| {name} | {value} |' for name, value in env.items())
    original_env = json.dumps(bundle.get('environment', {}), ensure_ascii=False, indent=2)
    source_table = '\n'.join(f'- `{name}`: `{digest}`' for name, digest in source_hashes.items())
    return f'''# 배터리 열폭주 6단계 분류 — 제출 코드

## 파일 구성

```text
model.py
train.py
test.py
requirements.txt
README.md
weights/
  model.joblib
```

`model.py`는 공통 피처·모델·예측 함수, `train.py`는 학습 및 가중치 저장, `test.py`는 가중치 로딩 및 제출 CSV 생성을 담당합니다. 대회 원본 데이터와 캐시는 제출 ZIP에 포함하지 않습니다.

## 실행 환경

- 제출 패키지를 확인한 OS: {os_description()}
- 아키텍처: {platform.machine()}
- 패키징/검증 Python: {platform.python_version()}
- 모델의 학습 실행 기록 Python: {bundle.get('environment', {}).get('python', '기록 없음')}
- 연산 장치: CPU. GPU 및 CUDA는 사용하지 않습니다.
- 모델 코드 버전: `{bundle['version']}`

패키징/검증 환경의 패키지 버전은 다음과 같으며 `requirements.txt`에 고정했습니다. PyTorch는 사용하지 않아 requirements에 포함하지 않았습니다.

| 패키지 | 버전 |
|---|---|
{package_table}

모델에 저장된 학습 당시 환경 정보:

```json
{original_env}
```

학습 당시 OS는 모델 파일에 별도로 기록되지 않았습니다. 위 OS는 이 패키지를 생성한 실행 환경에서 수집한 값입니다.

## 설치 및 추론

ZIP을 풀고 이 README가 있는 폴더에서 실행합니다. 데이터 경로는 실제 위치로 바꾸세요. Linux에서는 LightGBM이 사용하는 OpenMP 런타임이 필요합니다.

```bash
python -m pip install -r requirements.txt

python test.py \\
  --input ./data/battery_data_light/test/test.csv \\
  --image_root ./data/battery_data_light/test \\
  --model ./weights/model.joblib \\
  --sample_submission ./data/battery_data_light/sample_submission.csv \\
  --output ./submission.csv \\
  --n_jobs 4
```

`--sample_submission`을 주면 해당 sample_id 순서로 저장하고, 생략하면 입력 CSV 순서로 저장합니다. 출력은 `sample_id,label`이며 라벨은 1~6 정수입니다. 추론 과정에서 Test로 모델·전처리·확률 배율을 fit하지 않습니다.

## 가중치 및 전처리

{weight_location}

- 파일명: `model.joblib`
- 크기: {model_size:,} bytes ({model_size/1_000_000:.3f} MB)
- SHA-256: `{model_hash}`
- bundle 내용: 학습된 LightGBM Pipeline {len(classifiers)}개, 피처 목록과 순서, 클래스 확률 배율, 학습 설정, 코드 버전, 환경 정보

별도 스케일러·대치기·범주 인코더는 학습하지 않습니다. 센서/메타데이터는 수치 변환하고 결측은 LightGBM이 처리합니다. 이미지 밝기는 고정값 255로 나누며, 이미지 통계와 시계열 통계는 정해진 산식으로 계산합니다. 따라서 추가로 제출해야 할 학습된 전처리 파일은 없습니다.

25MB 초과 파일을 링크로 제공하는 경우 링크 접근 권한을 심사자가 다운로드할 수 있도록 설정해야 합니다. 다운로드 후 파일명을 `weights/model.joblib`로 맞추세요.

## 학습·검증 구성

- 최종 모델 학습 데이터: **{fit_data}**
- Train과 Validation의 experiment_id 및 sample_id가 겹치지 않는지 검사합니다.
- 시간 간격 증강 및 이미지 증강은 사용하지 않습니다.
- 클래스 가중치: `{cfg.get('class_weight', 'balanced')}`
- seed: `{seeds}`
- 피처 수: {len(bundle['features'])}
- 제외 피처 설정: `{cfg.get('drop', [])}`
- 검증: Train으로 학습한 모델을 제공 Validation에 적용합니다. 합산 재학습 옵션을 사용해도 보고서의 Validation 결과는 합산 재학습 전 모델에 대한 값입니다.
- OOF: `--cv` 또는 `--calibration oof` 사용 시 Train의 experiment_id 단위 GroupKFold를 수행합니다. fold 모델을 자동으로 제출 앙상블에 넣지는 않습니다.
- 여러 seed가 있으면 해당 모델의 클래스 확률을 평균합니다.
- {calibration_note}
- 저장된 클래스 배율: `{bundle['class_scale']}`

동일 설정으로 학습하는 명령:

```bash
{train_command}
```

LightGBM 주요 설정: n_estimators={params['n_estimators']}, learning_rate={params['learning_rate']}, num_leaves={params['num_leaves']}, min_child_samples={params['min_child_samples']}, colsample_bytree={params['colsample_bytree']}, subsample={params['subsample']}, subsample_freq={params['subsample_freq']}, reg_lambda={params['reg_lambda']}입니다. subsample_freq=0이면 행 bagging은 비활성입니다.

## 이미지·도메인 피처

이미지는 각 행의 image_path로 연결하고 회색조 0~1 밝기로 변환합니다. 평균·표준편차·최댓값·95/99 분위수, 고정 밝기 임계값 이상 픽셀 비율, 중앙 영역 통계, 중앙과 가장자리·상하·좌우 차이 등 16개 통계를 사용합니다. 이 값은 온도 단위가 아닌 밝기 통계입니다.

온도 차이·최댓값·최솟값·외기 대비 상승량, 가스 합과 비율, 전압 임계값, 히터 관측 여부 및 온도×가스·온도×이미지·온도 변화율×가스·온도 상승률×전압 하락률 교호작용을 계산합니다. 실제 배터리 단계의 의미를 정답에서 역산한 피처는 사용하지 않습니다.

## 시계열 피처 및 인과성

같은 experiment_id 내부에서 relative_time_s와 sample_id 순으로 정렬합니다. 현재 시각 t보다 이후의 관측은 사용하지 않습니다. 윈도우는 행 수가 아니라 **15·60·180·600초**, 구간은 **(t-window, t]**입니다. 같은 시각의 관측은 사용 가능합니다.

신호 15개: meas_v, meas_a, meas_t1, meas_t2, meas_p, meas_co, meas_co2, meas_ch4, meas_hf, meas_hcl, meas_hcn, img_max, img_p99, img_center_max, img_hot_085.

각 신호/윈도우에서 평균·최댓값·현재값과 윈도우 최솟값의 차이·윈도우 최댓값과 현재값의 차이·초당 변화율을 만듭니다. 변화율은 t-window 이하의 마지막 관측을 기준으로 실제 경과시간으로 나눕니다. 기준 관측이 없으면 결측입니다. 15초 변화율과 60초 변화율의 차이도 사용합니다.

가스 합과 HF의 누적합은 실험 시작부터 현재 시각까지 관측된 값만 더합니다. 관측값 누적합은 시간 적분이나 물리적 누적 발생량이 아니며 관측 빈도에 영향을 받습니다. 동일 시각의 누적 피처는 그 시각의 모든 관측을 포함합니다.

실험 전체 평균·최댓값·길이·순위, 다른 실험 관측, 미래 관측, 라벨 또는 라벨에서 파생한 값을 입력 피처로 사용하지 않습니다. 학습 라벨은 지도학습 및 명시한 모델 선택에만 사용합니다. 과거 관측을 삭제하거나 시간 기준을 바꾸면 시계열 피처는 달라질 수 있습니다.

## 사전학습 모델·외부 데이터

사전학습 모델과 외부 학습 데이터는 사용하지 않았습니다. LightGBM은 제공 데이터에서 새로 학습합니다. GPU 모델·PyTorch·CNN 임베딩은 사용하지 않습니다.

## 제출 재현 확인

{verification}

이 검사는 기존 제출과 예측이 같다는 확인이며, 정답에 대한 성능 평가나 대회 공식 검사를 대신하지 않습니다. 리더보드 점수는 이 스크립트가 조회하거나 검증하지 않습니다.

제출 소스 SHA-256:

{source_table}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('submission_v9'))
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--reference_submission', type=Path)
    parser.add_argument('--weights_url', help='HTTPS Drive download/share link if model is larger than 25 MB')
    parser.add_argument('--n_jobs', type=int, default=4)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    destination = args.output_dir.resolve()
    zip_path = destination.with_name(destination.name+'.zip')
    if destination.exists() or zip_path.exists():
        parser.error('Output directory/ZIP already exists. Choose another --output_dir; nothing was overwritten.')
    if args.reference_submission and not args.data_dir:
        parser.error('--reference_submission requires --data_dir')
    if args.weights_url:
        url = urlparse(args.weights_url)
        if url.scheme != 'https' or url.hostname not in ('drive.google.com', 'docs.google.com'):
            parser.error('--weights_url must be an HTTPS Google Drive link')
    if not args.model.is_file():
        parser.error(f'Model not found: {args.model.resolve()}')
    bundle = joblib.load(args.model)
    if bundle.get('version') != M.VERSION:
        parser.error('Model and source versions differ; use the v9 code matching your submitted model')
    for name in ['model.py', 'train.py', 'test.py']:
        compile((source/name).read_text(), str(source/name), 'exec')
    env = {name: version(name) for name in PACKAGES}
    verification = '이 패키징 실행에서는 기존 제출 CSV와 예측 일치 검사를 수행하지 않았습니다.'
    if args.reference_submission:
        verification = verify(bundle, args.data_dir, args.reference_submission, args.n_jobs)
    size = args.model.stat().st_size
    model_hash = sha256(args.model)
    source_hashes = {name: sha256(source/name) for name in ['model.py', 'train.py', 'test.py']}
    # Prepare a concrete package before checking whether a large-file URL is missing.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='submission-build-', dir=destination.parent) as temp:
        staging = Path(temp)
        for name in source_hashes:
            shutil.copy2(source/name, staging/name)
        (staging/'requirements.txt').write_text('\n'.join(f'{name}=={env[name]}' for name in PACKAGES)+'\n')
        (staging/'weights').mkdir()
        if size <= LIMIT_BYTES:
            shutil.copy2(args.model, staging/'weights/model.joblib')
        else:
            (staging/'weights/README.md').write_text('README.md의 다운로드 링크에서 모델을 받아 이 폴더에 model.joblib로 저장하세요.\n', encoding='utf-8')
        description = readme(bundle, env, model_hash, size, args.weights_url or 'DRIVE_LINK_REQUIRED', verification, source_hashes)
        (staging/'README.md').write_text(description, encoding='utf-8')
        if size > LIMIT_BYTES and not args.weights_url:
            draft = destination.with_name(destination.name+'_README_DRAFT.md')
            if draft.exists():
                parser.error('Draft path already exists; use another --output_dir')
            shutil.copy2(staging/'README.md', draft)
            raise SystemExit(f'Model is {size/1_000_000:.3f} MB (>25 MB). No final ZIP created.\n'
                             f'Prepared README draft: {draft}\n'
                             'Upload your model to Drive, then rerun with --weights_url "https://drive.google.com/...".\n'
                             'The upload is not performed by this script.')
        shutil.copytree(staging, destination)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(destination))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
    print(f'Final submission directory: {destination}')
    print(f'Final submission ZIP: {zip_path}')
    print(f'Model size: {size/1_000_000:.3f} MB; included in ZIP: {size <= LIMIT_BYTES}')
    if size > LIMIT_BYTES:
        print('Drive URL saved in README. Download availability/access permissions were not verified.')


if __name__ == '__main__':
    main()
