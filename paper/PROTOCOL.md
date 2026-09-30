# 실험 프로토콜 (canon_frame, 2026-09-15 ~ 09-18)

모든 파지 실험은 ICRA 논문의 Isaac 파이프라인(`icra/`) 위에서 수행했고, 완성 서버와 파지 생성기는 동결된 상태다.
바뀐 것은 완성 서버 앞에 붙는 **프레임 보정 한 단계**뿐이다.

## 1. 시뮬레이션 설정

- **시뮬레이터**: Isaac Sim 6.0, Franka Panda(7-DoF + 평행 그리퍼, 최대 개구 76.8 mm). 물체는 탁자 위 정지 상태,
  시행마다 (seed, trial)로 결정되는 yaw로 재배치 → **모든 arm이 동일한 자세 열을 본다**(paired 비교).
- **관측**: 고정 카메라 1대의 단일 시점 depth → 물체 점군 2,048점(bbox 정규화 후 서버 입력).
- **파지 생성/실행**: GraspGen(franka checkpoint) 후보 2,000개 → 기하 사전 필터(지지 여유, 접근각, 손끝 바닥,
  닫힘 기울기, antipodal/jaw span) → 신뢰도 최대 후보 실행. 모든 arm 동일.
- **성공 판정**: 물체 z-gain ≥ 0.10 m를 60 스텝 연속 유지(`LIFT_MIN_GAIN`, `LIFT_HOLD_STEPS`).
- **가림 모델(측면 가림)**: 관측 점군에서 카메라 오른쪽 축(`right = fwd × z`) 방향으로 가장 먼 점들을 비율만큼
  제거(`--obs-lateral-crop 0.25 | 0.4`; `icra/isaac_graspgen/indy7.py`). 두 네트워크가 학습에 쓴 반공간 crop과
  같은 형태의 결손. (깊이 껍질 절단 모델은 모든 arm을 ~28 %로 붕괴시켜 폐기 — RESULTS §5.)

## 2. 비교 arm

| arm | 입력 → 파지 계획 대상 |
|---|---|
| **plain completion** (`gate_off`) | 관측 점군을 bbox 중심/최대 반지름으로 정규화 → PoinTr+EDL 완성(6,144 생성 + 2,048 관측) |
| **partial only** (`partial_only`) | 관측 점군 2,048점 그대로 |
| **v2** (`complete_v2`) | 관측 점군 → bbox 정규화 → 회귀기가 예측한 중심 이동 Δc, log 스케일 s로 재정규화 → 동일 완성 |
| v1 (`reg_v1`, 참고) | 회귀기를 PoinTr식 무작위 crop으로 학습한 초기 버전 |

## 3. 물체군과 시행 수

- YCB 22개 자산 점검 → 파지 가능 16개(개구 77 mm보다 넓은 master chef can, tuna can, bowl, wood block와 상면
  후보가 없는 scissors 제외). **주 물체군 14개** = 16개 − 대형/특대형 클램프(ShapeNet-55에 없는 얇은 평면 도구;
  회귀기가 자신 있게 틀림 — 전 시야 66 → 14 %; 수치는 RESULTS에 보존).
- 셀당 **50 시행**(seed 0). 가림 3수준 × 3 arm × 16 물체 = 800 시행/arm/수준(주 14개: 700).
- 재현: seed 1에서 40 % 가림(원 물체 9개, 클램프 제외 8개)과 전 시야; v2는 seed 1에서 두 번 독립 추첨.
  같은 seed의 반복 실행은 실행 파지가 11 %만 일치(GraspGen 재샘플링) → 동일 자세에 대한 독립 추첨.
- 실행되지 않은 시행(후보 없음 등)은 실패로 계산: 성공률 = 성공 수 / (50 × 물체 수).

## 4. 통계

- 주 검정: **McNemar** (동일 (물체, 시행) 쌍에서 불일치 쌍 b/c, 정확 이항 양측 p). 보조: Fisher 정확검정, Wilson 95 % CI.
- 사전 결정: 주 물체군(14개)은 결과를 보고 정한 것이 아니라 클램프의 형상 부적합(ShapeNet 미포함) 근거로 제외,
  16개 전체 결과도 함께 보고.

## 5. 시뮬 도메인 벤치 (메커니즘)

- Isaac에 저장된 관측 점군(10 물체 × 15 관측, 전 시야와 40 % 가림)을 각 서버로 다시 완성 → YCB 메시 표면 샘플과의
  Chamfer 거리(mm), bbox 중심 오차(mm, 축별), 스케일 비. `canon_frame/analysis/sim_bench.py`.
- 오프라인 ShapeNet CD는 v1의 실패를 예측하지 못했고(CD −32 %인데 파지 −6.8 pp) 이 벤치는 예측했다 → 벤치만 사용.

## 6. 프레임 회귀기 v2

- 구조: PointNet(`CenterNet`, width 512), 입력 bbox 정규화 2,048점, 출력 중심 이동(3) + log 스케일(1). 1.4 M 파라미터.
- 학습 데이터: ShapeNet-55 train의 **hidden-point-removal 단일 시점**(형상당 3 시점, 고도 5–50°)
  + 측면 가림 증강(p 0.5, 가시점의 5–50 % 제거) + 무작위 yaw + 25 % PoinTr식 crop. 타깃 = 전체 형상의 중심/반지름.
- 손실 SmoothL1, Adam 1e-3 cosine, 40 epoch(~20 min). held-out 중심 오차 0.161 r(bbox) → 0.078 r.
- v1과의 차이는 학습 부분 점군뿐(반공간 crop → 단일 시점). v1은 실제 depth 뷰의 중심을 오히려 더 틀리게 했다
  (전 시야 10 → 31 mm): 병 양쪽 벽을 모두 보고 학습해 앞면만 보이는 센서 입력을 뒤로 늘렸기 때문.
- 배포: `canon_frame/engine/server_canon.py`(port 5560), ICRA 완성 서버와 같은 wire 규약; 추가 연산 ≈ 1 ms.

## 7. 진단 실험 (왜 프레임인가)

- yaw 민감도(`yaw_sensitivity.py`): held-out ShapeNet 500 구름 × 12 yaw × 3 seed. 무작위 yaw는 CD-L1 +27 %(최악 +40 %),
  평균 epistemic ×8.6(r 0.96). GT 프레임에서 epistemic argmin이 손실의 91–98 % 회복, **서버(bbox 정규화) 경유 시 0 %**.
- 정규화 방식(`norm_yaw_offline.py`): 100 구름 × 8 yaw. bbox 중심(오차 0.20 ± 0.13 r)은 CD-L1 +75 %, 무게중심 +124 %;
  스케일 오차는 무해; bbox 오프셋은 형상 의존(고정 시선 방향 이동으로 회복 0) → 학습이 필요.

## 8. 전이 검증 프로토콜 (SeedFormer)

- 모델: github.com/hrzhou2/seedformer 공식 ShapeNet-55 checkpoint(저자 Google Drive 폴더 `1waTq7npTO068qyOAkK7HkSW0z69JVwAE`,
  `models/shapenet55/ShapeNet-55/ckpt-best.pth`, epoch 255, `seedformer_dim128`, `UPSAMPLE_FACTORS [1,4,4]`, 2048 → 8192점).
  학습 규약(저자 코드 `utils/data_loaders.py::pc_norm`, `CONST.N_INPUT_POINTS 2048`): 전체 형상의 평균 중심·최대 반지름 정규화 후 2048점 crop —
  PoinTr의 ShapeNet-55 로더와 동일하고, 회귀기 v2의 타깃 프레임(`targets()`: 중심 0 / 반지름 1)과 같음 → 회귀기·완성 네트워크 모두 고정, 무수정.
- 입력: §5의 저장 관측(10 물체 × 15 × {전 시야, 40 %}, 파일 목록 `results/seedformer_bench/meta.json`)의 2048점 tail, 세 조건 모두 동일 점.
  up-axis 회전은 완성 서버와 동일(`R_align`, 세계 z-up → ShapeNet y-up).
- 조건: bbox(부분 점군 bbox 중심 + 최대 반지름) / v2(bbox 프레임 → 회귀기의 중심 이동·log 스케일 적용, 재정규화 없음) / gt(메시 평균 중심·최대
  반지름, 진단용). 출력은 들어간 프레임으로 역변환. `torch.manual_seed(0)` 후 forward; SeedFormer 내부 FPS는 입력에 대해 결정적.
- 평가: 8192 출력점 전부(서버가 덧붙이는 관측 tail 없음; 네트워크가 관측 영역을 스스로 재생성 — 출력의 ~26 %가 입력점 0.1 mm 이내, 최소거리
  중앙값 0.3 µm) vs 메시 8192 샘플, CD-L1·precision·recall(mm), 프레임 중심 오차·스케일 비, 시간(`cuda.synchronize`).
- 스크립트 `analysis/sim_bench_seedformer.py --per-cell 15`; 물체 제외·튜닝 없음.

## 9. 재현 명령 (PoinTr 루트에서)

```
python canon_frame/canon/precompute_visibility.py --subset train --views 3      # 8 min
python canon_frame/canon/train_center_reg_v2.py --epochs 40 --width 512          # GPU 20 min
# 서버: canon_frame/engine/server_canon.py (port 5560)
python canon_frame/analysis/sim_bench.py --tag runLat04 --arm gate_off --servers base=172.17.0.2:5557 v2=127.0.0.1:5560
SEED=0 STAGE=both bash canon_frame/run_isaac.sh canon_frame/ckpts/center_reg_v2.pth complete_v2 5560
bash canon_frame/tools/morning_wrapup.sh                                          # 표 생성
python canon_frame/paper/figures/make_figures.py                                               # 이 패키지의 그림 (컨테이너)
```
