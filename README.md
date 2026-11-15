# Uni-D 배터리 열폭주 6단계 분류


## 팀 정보
- 팀명: 9팀
- 최종 선택 제출 파일: submission_v9_seeds.csv
- model, test, train 파일 위치: song/battery_v9

## 개발 환경
- OS: Ubuntu 22.04
- Python: 3.10.12
- GPU: 대회 제공 GPU 서버
- 라이브러리: `requirements.txt` 참고

## 파일 구성
| 파일 | 역할 |
|---|---|
| `model.py` | 도메인·과거 방향 시계열 피처 생성(create_fast_features), LightGBM 모델 및 시계열 스무딩/클래스 가중치 후처리 정의 |
| `train.py` | Train+Validation 통합 데이터 기반 실험 단위 GroupKFold(5) 교차검증, 5개 폴드 모델 및 최적 후처리 가중치(best_weights) 저장 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 피처 생성 및 5-Fold 앙상블 추론·후처리 후 제출 파일 생성성 |
| `model.joblib` | 학습된 5-Fold LightGBM 모델 리스트, 피처 컬럼 목록, 최적 클래스 가중치(best_weights)가 포함된 아티팩트. 재학습 없이 test.py만으로 결과를 재현할 수 있습니다 |

## 실행 방법
```bash
pip install -r requirements.txt

# 학습 (선택: --refit_train_val 을 주면 Train+Validation으로 최종 학습)
python train.py --data_dir <battery_data_light 경로> --model_out model.joblib

# 추론
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --model model.joblib --output submission.csv \
               --sample_submission <battery_data_light 경로>/sample_submission.csv
```

## 학습·검증 구성
- 학습 데이터: Train+Validation
- 검증 방법: 실험 단위(experiment_id) 기준 GroupKFold(5) 교차검증 및 OOF(Out-Of-Fold) 예측 기반 Macro F1 최적화
- 검증 점수: <Validation Macro F1>

## 시계열 피처 구성
- 결측·상수 노이즈 컬럼(indu_t, indu_t.1, meas_a)을 제거하고 물리·화학 도메인 조합 피처(셀 간 온도 편차 |meas_t1 - meas_t2|, 주변 온도 대비 상승분 meas_t1 - atmp_t, 10종 가스 총합 total_gas, 불완전 연소 비율 meas_co / meas_co2)를 생성했습니다.
- 같은 experiment_id 안에서 relative_time_s 기준 오름차순 정렬 후, 핵심 센서(meas_t1, meas_t2, thermal_t, meas_v, meas_p, total_gas, meas_co)에 대해 직전 1스텝·5스텝 대비 차분(diff(1), diff(5))과 과거 30개 관측 구간의 이동 평균·이동 표준편차 및 현재값과의 차이(current - rolling_mean)를 추출했습니다.
- 미래 시점 관측과 실험 전체 기준 통계는 사용하지 않습니다.

## 사용한 사전학습 모델
- 없음 (LightGBM 기반 탭룰러·시계열 모델만 사용)
