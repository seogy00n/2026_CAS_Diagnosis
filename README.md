# 갑상선 안병증(TAO) CAS 항목별 자동 분류

안구 사진만으로 갑상선 안병증의 임상 활성도 점수(Clinical Activity Score, CAS)
항목별 정상/비정상을 자동 분류하는 딥러닝 모델. 영남대학교병원 안과 임상 협력.

- **논문** 딥러닝 기반 안구 사진을 이용한 갑상선 안병증(TAO)의 임상 활성도 점수(CAS) 항목별 자동 분류
- **기간** 2026.03 ~ 진행 중
- **과목** 전자공학종합설계 (졸업 프로젝트)

---

## ⚠️ 데이터 정책 — 먼저 읽을 것

**이 저장소에는 환자 데이터가 일절 포함되어 있지 않으며, 앞으로도 포함되지 않는다.**

원본 데이터는 환자 안면·안구 사진과 실명이 포함된 정답지로 구성된 임상 데이터다.
IRB 승인 범위를 벗어난 제3자 서버 전송에 해당하므로 다음은 **절대 커밋하지 않는다.**

| 대상 | 사유 |
|---|---|
| 안구·안면 사진 전체 | 신원 식별 가능한 생체 의료영상 |
| 정답지 `.csv` / `.xlsx` | 환자 실명 및 차트번호 포함 |
| 대상자 명단 `.xlsx` | 실명 명단 |
| 특징 캐시 `.npy` | 환자 영상에서 추출된 파생 데이터 |

[`.gitignore`](.gitignore)는 **deny-all + allowlist** 방식이다.
루트의 모든 것을 무시한 뒤 코드와 문서만 되살리므로, 새로 추가되는 파일은
기본적으로 추적되지 않는다. 이미지·표 형식 파일 확장자는 3단계 이중 차단이 걸려 있다.

> 규칙을 완화하기 전에 반드시 개인정보 포함 여부를 확인할 것.

---

## 저장소 구조

```
backbone_exp/
  casexp.py      데이터 로딩 · 백본 레지스트리 · 특징 추출 · 헤드 정의
  run.py         환자 단위 5-겹 교차검증 실행기 (CLI)
  results.json   실행 결과 (수치만)
  cache/         특징 캐시 — git 추적 제외
docs/            연구 정리본
```

로컬 데이터는 다음 경로를 가정한다 (저장소에는 없음).

```
실습/data_origin/<환자>/{left,right}/*.jpg     정제 영상 294장
TED_Data/<환자>/<CAS_n>/{normal,abnormal}/      항목별 라벨 (폴더 구조가 곧 라벨)
CAS_Union_Ground_Truth.xlsx                     평가자별 정답지 (20명분)
```

---

## 설계 원칙

1. **백본은 항상 동결한다.** 특징을 한 번 뽑아 캐시하고 헤드만 재학습한다.
   환자 31명 규모에서 8,740만 파라미터를 미세조정하면 과적합된다(Phase 3에서 확인).
2. **분할은 환자 단위로 고정한다.** 라벨이 환자 단위로 붙어 같은 환자의 영상 전체에
   복사되므로, 영상 단위 분할은 학습·검증 간 라벨 누수를 일으킨다.
   모든 백본이 동일 시드의 동일 fold를 공유한다.
3. **임계값은 학습 fold 안에서만 고른다.** 학습 fold 내부에서 다시 교차검증해
   표본 외 예측을 만들고 거기서 임계값을 선택한다. 검증 fold는 평가에만 쓴다.
4. **재현율을 우선한다.** 자가 선별 도구이므로 위음성이 위양성보다 위험하다.
   임계값 규칙은 "재현율 ≥ 목표치 중 정밀도 최대"이며, 불균형에 정직한 AUPRC를 병기한다.

---

## 실행

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install scikit-learn iterative-stratification timm openpyxl

# 백본 비교 (CPU만으로 동작)
python backbone_exp/run.py --backbones resnet18 resnet50 dinov2_s dinov2_b

# 선형 프로브
python backbone_exp/run.py --backbones dinov2_b --hidden 0

# Union 라벨로 민감도 분석 (20명분만 존재)
python backbone_exp/run.py --backbones dinov2_b --label-source union
```

GPU는 필요 없다. 특징 추출은 ResNet 약 24초, DINOv2 ViT-B/14 약 15분(CPU 4스레드).
캐시가 있으면 재실행 시 특징 추출을 건너뛴다.

---

## 결과

### 백본 비교 — 동일 fold · 동일 헤드 · 동일 임계값 규칙

| 백본 | 특징 차원 | OOF 거시 AUROC | OOF 거시 AUPRC | fold 편차 |
|---|---:|---:|---:|---:|
| ResNet-18 (ImageNet) | 512 | 0.5064 | 0.3647 | ± 0.121 |
| ResNet-50 (ImageNet) | 2048 | 0.5839 | 0.4209 | ± 0.080 |
| DINOv2 ViT-S/14 | 384 | 0.5733 | 0.4212 | ± 0.084 |
| **DINOv2 ViT-B/14** | 768 | **0.6149** | **0.4963** | **± 0.066** |

채택 모델(ViT-B/14)이 전 지표 1위이며 fold 편차도 가장 작다.
논문 발표값 OOF 0.6003에 대해 서로 다른 시드·다른 fold 분할에서 0.6149가 재현되었다.

### 항목별 OOF AUROC

| 항목 | ResNet-18 | ResNet-50 | ViT-S/14 | ViT-B/14 | 양성 |
|---|---:|---:|---:|---:|---:|
| CAS 3 안검 발적 | 0.4353 | 0.6029 | 0.6775 | **0.8350** | 60/294 |
| CAS 4 결막 발적 | 0.6980 | 0.7430 | 0.6743 | 0.7395 | 118/294 |
| CAS 5 안검 부종 | 0.4159 | 0.4571 | 0.5107 | 0.5471 | 180/294 |
| CAS 7 눈물언덕 염증 | 0.4762 | **0.5327** | 0.4306 | 0.3379 | 52/294 |

- **CAS 3은 백본이 지배한다.** AUROC 0.435 → 0.835, AUPRC 0.184 → 0.594.
- **CAS 7은 백본이 좋아질수록 나빠진다.** 헤드 4개가 동일한 전역 요약 벡터 하나를
  공유하는 현재 구조에서, 사진 면적의 1~2%인 눈물언덕이 묻히는 것으로 보인다.
  단 비정상 환자가 6명뿐이라 통계적 신뢰도가 낮다 — 가설이며 아직 증명은 아니다.
- CAS 6(결막 부종)은 수집 데이터 31명 전원에서 비정상 사례가 0건이라 학습 대상에서 제외했다.

---

## 미해결 이슈

| # | 내용 |
|---|---|
| 1 | `TED_Data` 라벨이 논문 기술(Union)과 불일치. 20명 기준 Union과 3명, 손 교수 판정과 1명 차이 |
| 2 | 논문의 "합성 영상 2,276장"은 학습셋 전체 크기. 실제 GAN 생성분은 500장 |
| 3 | ~~CNN 기준선 미측정~~ → 백본 비교로 해소 |
| 4 | 평가자별 원본 정답지가 20명분만 확보됨 (항목별 라벨 자체는 31명 전원 존재) |
| 5 | GAN 모델의 구체적 종류가 논문에 미기술 |

## 다음 작업

- **항목별 ROI 분리** — ViT 패치 토큰(518×518 → 37×37 = 1,369개)을 ROI로 풀링해
  순전파 한 번으로 부위별 특징을 얻는다. CAS 7 역전 현상의 검증 실험.
- 선형 프로브 및 색 통계 기준선 (ROI 내 Lab a\* 평균 + 로지스틱 회귀)
- 추가 확보 데이터의 항목별 라벨 확충 — 총점만 있고 항목별 판정이 없다

---

## 참고 문헌

- M. P. Mourits et al., "Clinical criteria for the assessment of disease activity in Graves' ophthalmopathy," *Br J Ophthalmol*, 73:639–644, 1989.
- M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," *TMLR*, 2024.
- T. Darcet et al., "Vision Transformers Need Registers," *ICLR*, 2024.
- K. He et al., "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
