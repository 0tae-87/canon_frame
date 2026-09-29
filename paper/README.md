# IPIU paper package — 관측 프레임 보정으로 가려진 물체의 완성 기반 파지 성공률 높이기

`canon_frame` 연구(2026-09-15 ~ 09-18, 종료)를 국내 학술대회(IPIU) 논문용으로 정리한 폴더 — 같은 repo의
`paper/`에 둔다(연구 원본 코드·체크포인트·raw 결과는 상위 폴더). 연구 자체는 **완료**된 상태이며(최종 방법 v2,
3개 결과표, seed 재현, 메커니즘 벤치, 닫힌 부정 결과 9건), 이 폴더는 논문 작성에 필요한 **프로토콜·결과·그림**을
한곳에 모은 것이다. 모든 숫자는 `../results/`의 파일에서 `figures/make_figures.py`로 재계산된다(원 표
`../results/*.md`와 소수점까지 일치).

| 파일 | 내용 |
|---|---|
| [PAPER_OUTLINE.md](PAPER_OUTLINE.md) | 논문 구성안: 제목·초록·절별 핵심 문장·그림/표 배치·한계 |
| [PROTOCOL.md](PROTOCOL.md) | 실험 프로토콜 전문: 시뮬레이터, 물체, 가림 모델, arm 정의, 시행 수, 통계, 회귀기 학습 |
| [RESULTS.md](RESULTS.md) | 모든 결과표(Isaac 파지, 벤치 CD, 진단, 부정 결과)와 각 숫자의 데이터 출처 |
| `figures/` | 논문 그림 7장(PNG+PDF) + 생성 스크립트 `make_figures.py` + 장면 이미지 |
| `../results/` | 원천 데이터(상위 폴더): 실행된 시행 행 `exec_rows/*.csv`(논문 표는 12개 태그 사용), 벤치 로그, 진단 JSON, 학습 로그 |

## 한 문단 요약 (논문 주장)

완성 서버는 관측된 부분 점군을 그 점군의 bounding box 중심과 최대 반지름으로 정규화하지만, 완성 네트워크는
물체 전체의 프레임에 놓인 부분 점군으로 학습되었다. 물체의 한쪽이 가려지면 이 프레임은 틀리고(중심 오차
≈ 0.2 물체 반지름), 완성 결과가 엉뚱한 위치에 놓여 **완성이 파지에 오히려 해가 된다**(40 % 측면 가림: 완성
41 % vs 부분 점군만 47 %). ShapeNet-55의 카메라형 단일 시점 부분 점군으로 학습한 1.4 M 파라미터 PointNet
회귀기가 부분 점군만으로 올바른 중심과 스케일을 예측하며, 완성 전에 이것으로 재정규화하면 완성이 다시 부분
점군보다 좋아지고(52 %), 그 차이는 가림이 심할수록 커지며(+3.2 → +4.0 → +5.4 pp; 주 물체군 14개에서는
+5.4 → +7.1 → +9.1 pp, 모두 p < 0.003), 완전 시야에서는 비용이 없다.

## 그림 목록

| 그림 | 파일 | 내용 | 데이터 |
|---|---|---|---|
| Fig. 1 | `fig0_pipeline` | 파이프라인 개요: 단일 시점 → bbox 정규화 → **프레임 회귀기** → 재정규화 → PoinTr+EDL(frozen) → GraspGen | — |
| Fig. 2 | `fig1_occlusion_trend` | 가림 정도(0/25/40 %)에 따른 파지 성공률: 완성(기존 프레임)/부분 점군/v2, 16개·주 14개 물체, Wilson 95 % CI, McNemar p | `exec_rows` |
| Fig. 3 | `fig2_per_object_lat40` | 40 % 가림, 물체별 성공률(주 14개) | `exec_rows` |
| Fig. 4 | `fig3_bench_mechanism` | 메커니즘: 시뮬 벤치의 완성–메시 CD, bbox 중심 오차, 축별 중심 오차 (plain / v1 / v2) | `sim_bench_v2.log` |
| Fig. 5 | `fig4_diagnosis_yaw_vs_centre` | 진단: yaw 오프셋 vs CD·epistemic(r 0.96), 중심 정규화 방식별 CD 손실(+75 % / +124 %), bbox 중심 오차 분포 | `yaw_sensitivity_A_summary.json`, `norm_yaw_offline.json` |
| Fig. 6 | `fig5_seed_replication` | 40 % 가림, seed 0 / seed 1 / seed 1 재추첨의 독립 재현 (8개 물체) | `exec_rows` |
| Fig. 7 | `fig6_regressor_training` | 회귀기 v2 학습 곡선: held-out 중심 오차 0.161 r(bbox) → 0.078 r | `center_reg_v2_train.log` |
| 보조 | `scene_objects_topview.png` | Isaac 장면의 YCB 물체 상면도 | — |

그림 재생성: `docker run --rm -v /home/wim/Desktop/yt_ws:/workspace -w /workspace/PoinTr pointr_blackwell:gpufix python canon_frame/paper/figures/make_figures.py`

## 남은 일 (연구가 아니라 집필)

- 연구 항목은 없음. 09-18 종결 판정: 프레임이 맞고 나면 완성 CD를 더 줄여도(−10 %까지) 파지 성공률은 변하지
  않았고, 파지를 움직인 유일한 변화는 v1 → v2(학습 부분 점군을 센서와 일치시킨 것)였다.
- 논문에 정직하게 적어야 할 한계(PAPER_OUTLINE §한계): 시뮬레이션 전용, 완전 시야에서는 v2 − plain이 음수(−3.5 pp,
  n.s.), 얇고 납작한 도구(클램프)는 ShapeNet-55에 없는 형상이라 회귀기가 자신 있게 틀림(주 물체군에서 제외, 수치는
  보존), seed 재현은 40 % 가림·전 시야만(25 %는 seed 0만).
- 선택 사항(GPU 없이 가능): 정성 그림 — 같은 관측에 대한 plain / v2 완성 결과와 실행된 파지 위치 비교
  (`canon_frame/analysis/sim_paired_geometry.py`의 덤프 필요; 덤프는 `icra/isaac_graspgen/output/graspgen/`에 있으나 git에는 없음).
