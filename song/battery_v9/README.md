# v9 — try1.ipynb의 높은 점수 구성 재현

사용자가 리더보드 0.89대를 확인한 `try1.ipynb`의 마지막 `work/submission.csv` 생성 경로를 기준으로 작성했습니다. 실제 대회 데이터를 실행한 결과가 아니므로 같은 리더보드 점수나 0.9 달성을 보장하지 않습니다. 첫 목적은 v8의 설정을 더 변경하는 것이 아니라, 성능이 확인된 노트북 구성을 명시적인 스크립트로 재현하는 것입니다.

## 먼저 알아둘 점

- 노트북의 최종 제출 함수는 `model.predict(...)`를 사용합니다. 계산된 `best_weights`는 그 제출 경로에 적용되지 않습니다. 사용자도 마지막 셀의 제출 파일이라고 확인했습니다.
- 실제 모델은 balanced LightGBM입니다. 상단의 로지스틱 회귀 설명은 실제 구현과 다릅니다.
- 마지막 경로에는 시간 간격 증강, RGB 피처, seed/fold 앙상블, Train+Validation 합산 학습이 없습니다.
- 기존 0.89대 CSV와 노트북을 덮어쓰지 마세요. 새 버전은 별도 폴더에서 실행합니다.

## 1. 배치·설치

압축 안의 `battery_v9` 폴더 전체를 서버 `/home/ubuntu/song/`에 업로드합니다. 기존 `uni_d_baseline_v7`의 세 파일과 섞지 않습니다.

```bash
cd /home/ubuntu/song/battery_v9
python -m pip install -r requirements.txt
python -m py_compile model.py train.py test.py checks.py check_model.py
python -c "import model; print(model.__file__); print(model.VERSION)"
```

버전 출력은 `battery-v9-try1.1`이어야 합니다. 가능하면 노트북을 실행했던 가상환경을 그대로 사용하세요. 학습·추론의 패키지 버전도 맞추세요.

## 2. 노트북 구성으로 학습

```bash
python train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --model_out ./weights/v9_try1.joblib \
  --report_dir ./reports/v9_try1 \
  --n_jobs 4
```

기본값:

| 항목 | 값 |
|---|---|
| 피처 | 377개, 노트북과 같은 순서 |
| 학습 데이터 | Train만 |
| 클래스 가중치 | balanced |
| seed | 42 |
| LightGBM | 600 trees, learning_rate 0.03, num_leaves 31 |
| min_child_samples | 30 |
| colsample_bytree | 0.8 |
| subsample_for_bin | 50000 |
| subsample / subsample_freq | 0.8 / 0 |
| reg_alpha / reg_lambda | 0 / 0 |
| 증강 | 없음 |
| 확률 배율·시간 후처리 | 없음 |

노트북에서는 `subsample_freq`가 지정되지 않아 기본값 0이므로 행 bagging이 활성화되지 않았습니다. 재현을 위해 이 동작을 유지합니다. CPU 코어 수는 `--n_jobs`로 조절하며, 노트북과 같이 전부 사용하려면 `--n_jobs -1`을 지정할 수 있습니다.

기본 실행은 OOF를 반복하지 않고 Train 모델 한 번을 학습합니다. 따라서 v8 기본 실행보다 학습 횟수가 적습니다. `--cv`를 추가하면 별도로 Train의 GroupKFold를 평가하지만, fold 모델을 제출 앙상블로 사용하지는 않습니다.

학습 로그의 피처 수가 377인지 확인하세요. 옵션으로 피처를 제거한 경우에는 최종 사용 피처 수가 줄어듭니다.

## 3. 제출 CSV 생성

```bash
python test.py \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --model ./weights/v9_try1.joblib \
  --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output ./work/submission_v9_try1.csv \
  --n_jobs 4
```

새 제출 파일:

```text
/home/ubuntu/song/battery_v9/work/submission_v9_try1.csv
```

v7/v8의 `joblib` 파일은 호환되지 않습니다. v9로 새로 학습한 파일을 사용하세요. 제출 파일 생성 명령은 학습을 하지 않습니다.

0.89대 원본 CSV를 보관하고 있다면, 위 명령에 `--reference_submission 원본CSV의실제경로`를 추가할 수 있습니다. 두 파일의 sample_id를 맞추어 달라진 예측 개수·비율을 출력합니다. 원본 CSV는 읽기만 하며 학습/보정에는 사용하지 않습니다. 파일 경로를 출력 경로와 다르게 지정하세요. 예측 일치율은 정답에 대한 정확도가 아닙니다.

## v8과 무엇이 다른가

1. 15·60·180·600초 윈도우를 사용합니다. 각 신호의 윈도우 평균·최댓값·최솟값 대비 상승량·최댓값 대비 하락량·변화율, 15초와 60초 변화율 차이를 포함합니다.
2. `meas_hcl`, `meas_hcn`과 이미지 중앙 최댓값·밝기 0.85 이상 비율 등 노트북의 15개 신호를 사용합니다.
3. 온도 상승×가스, 온도×이미지, 온도 변화율×HF, 온도 상승률×전압 하락률 등 교호작용을 복원합니다.
4. 이미지 통계는 노트북의 회색조 16개입니다. RGB·HSV·3×3 격자 피처를 추가하지 않습니다. 밝기는 실제 온도 단위가 아닙니다.
5. 메타데이터는 노트북처럼 수치형으로 사용합니다. `relative_time_s`와 `indu_t.1`도 유지합니다. 실제 Test에서 전부 결측인 컬럼이 있더라도 첫 재현에서는 임의로 삭제하지 않습니다.
6. 누적 가스 피처는 노트북의 관측값 누적합을 유지합니다. 이것은 시간 적분이 아니며 관측 빈도에 민감합니다. 그렇지만 이를 임의로 시간 적분으로 바꾸면 원본 모델과 다른 실험이 되므로 기본값에서는 바꾸지 않았습니다.
7. 증강과 그에 따른 샘플 가중치, 범주 인코딩 변경, 추가 정규화를 제거합니다.

이 중 어느 변경이 리더보드 차이를 만든 원인인지는 아직 분리 검증하지 않았습니다. 많은 변경을 한꺼번에 한 v8의 결과만으로 특정 피처가 원인이라고 단정할 수 없습니다.

## 노트북에서 확인한 재현 문제와 수정 범위

- 이미지 캐시 셀은 `ccache`에 경로를 할당하고 `cache`를 참조합니다. 현재 커널의 숨은 변수에 의존할 수 있으므로 새 스크립트는 독립적인 joblib 캐시를 사용합니다.
- 저장된 셀 실행 순서가 위에서 아래 순서와 다릅니다. Validation 0.8013 출력의 실행 번호는 53이고, balanced 재학습 셀은 89입니다. 따라서 0.8013을 마지막 제출 모델의 검증 점수라고 볼 수 없습니다. 재학습 셀의 저장된 raw Validation 출력은 0.8587입니다.
- Validation 0.9111은 같은 Validation으로 확률 배율을 고른 뒤 측정한 선택 점수입니다. 독립 평가가 아니며, 그 배율은 확인된 최종 제출 경로에도 쓰이지 않습니다.
- 같은 시각이 여러 행이면 `sample_id`로 정렬을 고정합니다. 누적합은 그 시각의 모든 관측을 포함하도록 처리합니다. 시각이 고유한 경우 노트북과 같은 피처이며, 중복 시각의 경우에는 원본의 행 순서 의존성을 보완한 동작입니다.
- 선택적 원시 컬럼이 아예 없으면 NaN으로 채워 377개 스키마를 유지합니다. 무한대는 최종 피처에서 NaN으로 바꿉니다. 원본의 정상적인 유한값 데이터에는 영향을 주지 않습니다.
- 이미지 파일명 fallback에 여러 파일이 매칭되면 임의 선택하지 않고 오류를 냅니다.

따라서 노트북의 현재 소스와 정리된 코드가 같은 구성이라는 점과, 과거 커널에서 생성한 실제 제출 파일이 완전히 같다는 점은 구분해야 합니다. 과거 커널 상태·캐시·패키지 버전 차이는 기존 CSV와의 비교로 확인하세요. 피처를 새로 추출하므로 오래된 노트북 캐시는 사용하지 않습니다.

## 인과성 검사

```bash
python checks.py --smoke

python check_model.py \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image_root /home/ubuntu/data/battery_data_light/test \
  --model ./weights/v9_try1.joblib \
  --n_jobs 4 \
  --require_image_effect
```

`checks.py`는 합성 데이터만 사용합니다. `check_model.py`는 사용자 서버에서 실제 입력을 읽고, 행 순서 변경·미래 관측 제거·실험별 분리 시 피처와 예측이 유지되는지 검사합니다. 이미지 피처는 샘플별 독립 계산값을 재사용하고 시계열 피처는 매번 다시 만듭니다. 검은 이미지로 바꿨을 때 예측 변화율도 출력합니다. `--require_image_effect`를 주면 변화가 없을 때 오류를 내며, 노트북에 있던 이미지 민감도 검사를 확인하는 용도입니다. 실제 이미지 사용 여부에 대한 공식 검사는 대회 제공 검사 도구를 따르세요.

현재보다 이후의 관측이나 실험 전체 통계를 피처에 사용하지 않습니다. 한 실험의 과거 행까지 삭제했을 때, 같은 시각 행 일부를 삭제했을 때, 상대 시간을 다시 정의했을 때 예측이 유지되는 것은 보장하지 않습니다.

## 선택적 후속 실험 — 첫 재현 결과 확인 후

먼저 기본 모델의 실제 제출 점수를 확인하세요. 아래 설정은 기본 재현과 다른 실험이므로 별도 파일명으로 저장해야 합니다.

### A. seed 앙상블

```bash
python train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --seeds 42 2026 3407 \
  --model_out ./weights/v9_seeds.joblib \
  --report_dir ./reports/v9_seeds \
  --n_jobs 4
```

각 seed의 Train 모델 예측 확률을 평균합니다. 개선을 보장하지 않습니다.

### B. 확률 배율 비교

`--calibration validation`은 노트북의 bounded Nelder–Mead 탐색처럼 0.03~2.0 범위에서 양수 배율을 선택합니다. 최적화 결과가 무보정보다 좋아지지 않으면 배율을 모두 1로 유지합니다. 보정 결과는 모델 파일에 저장되어 `test.py`에 일관되게 적용됩니다.

```bash
python train.py \
  --data_dir /home/ubuntu/data/battery_data_light \
  --calibration validation \
  --model_out ./weights/v9_valscale.joblib \
  --report_dir ./reports/v9_valscale \
  --n_jobs 4
```

이때 보정 후 Validation 점수는 **모델 선택 점수**입니다. 독립 성능 개선의 근거로 해석하면 안 됩니다. `--calibration oof`를 쓰면 Train GroupKFold OOF로 배율을 고르고 별도 Validation에서 확인합니다. OOF 보정 후 점수 역시 선택 점수입니다. HMM이나 단계 최댓값 강제는 어느 모드에도 없습니다.

보정된 모델에서 무보정 예측을 비교하려면 `test.py --raw`를 사용합니다. 기본 `--calibration none` 모델에서는 이 옵션을 붙여도 예측이 같습니다. 노트북 출력의 반올림된 `0.000`을 그대로 복사해 특정 클래스 배율을 0으로 만들지 마세요.

### C. 추가 데이터로 재학습

`--refit_train_val`은 평가 후 Train+Validation으로 최종 모델을 다시 학습하는 별도 실험입니다. 노트북 재현 기본 경로에는 포함하지 않았습니다. Validation의 최종 학습 사용이 허용되는 범위에서 사용하세요. 보고서는 항상 합산 재학습 전 Train-only 모델의 Validation 결과를 나타냅니다. 합산 모델에 기존 보정을 적용하면 확률 분포가 바뀔 수 있으므로 원래 점수가 그대로 유지된다고 볼 수 없습니다.

## 보고서

- `summary.csv`: Validation raw/selected, 선택적 OOF 지표와 클래스별 F1
- `validation_confusion_raw.csv`, `validation_confusion_selected.csv`: 행=실제, 열=예측
- `metrics.json`: 전체 지표·학습 설정·패키지 버전·피처 목록
- `feature_importance.csv`: Train 모델의 gain 중요도

기본 모드에서는 raw와 selected가 동일합니다. 실제 원본 행이나 실험 ID를 보고서에 내보내지 않습니다. `metrics.json`에는 실행 경로 등 설정이 포함됩니다.

## 로컬 검증 결과

합성 데이터에서 노트북의 검토된 모델 정의 셀과 직접 비교했습니다.

- 고유 시각의 규칙적 간격: 377개 피처와 순서가 정확히 일치
- 고유 시각의 불규칙 간격: 377개 피처가 정확히 일치
- 센서 결측이 있는 입력: 377개 피처가 정확히 일치
- 같은 합성 데이터에서 원본과 새 코드의 600-tree LightGBM 예측 확률이 정확히 일치
- 이미지 통계 16개가 정확히 일치
- 별도 인과성·스키마 검사 7개 통과
- 실제 LightGBM 학습·저장·로딩·제출 생성·선택적 CV/보정/재학습 검사 통과

검사 환경: Python 3.9, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, LightGBM 4.6.0, Pillow 11.3.0. 실제 대회 데이터와 리더보드 성능은 이 환경에서 검증하지 않았습니다.
