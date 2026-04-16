# dual_viewer 프로젝트

gsplat(nerfstudio) 기반 3D Gaussian Splatting 뷰어. SIBR viewer 호환 포맷을 사용하며, 두 모델을 나란히 비교하는 듀얼 뷰를 지원한다.

## 사용법

```bash
cd examples

# 단일 뷰
python sibr_viewer.py -m /path/to/model

# 듀얼 뷰 (좌우 비교)
python sibr_viewer.py -m /path/to/model_A /path/to/model_B

# COLMAP 소스 직접 지정
python sibr_viewer.py -m /path/to/model_A /path/to/model_B -s /path/to/colmap

# Photometric refinement 비활성화
python sibr_viewer.py -m /path/to/model_A /path/to/model_B --refine 0

# 리파인 이터레이션/해상도 조정
python sibr_viewer.py -m /path/to/model_A /path/to/model_B --refine_iters 1000 --refine_downscale 2

# Coarse-to-fine 비활성화 (단일 해상도로 리파인)
python sibr_viewer.py -m /path/to/model_A /path/to/model_B --refine_no_ctf

# Early stopping patience 조정 (0=비활성화)
python sibr_viewer.py -m /path/to/model_A /path/to/model_B --refine_patience 100
```

브라우저에서 `localhost:8080` 접속. 마우스/키보드 조작 시 두 뷰가 동시에 움직인다.

## 입력 포맷 (SIBR 호환)

```
model_path/                          # -m 플래그
├── cameras.json                     # 카메라 파라미터 (COLMAP 없을 때 사용)
└── point_cloud/
    └── iteration_30000/
        └── point_cloud.ply          # 표준 3DGS PLY

source_path/                         # -s 플래그 (선택)
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

카메라 로딩 우선순위: `-s` COLMAP sparse > `model_path/cameras.json`

## 프로젝트 구조

### `examples/sibr_viewer.py` — 메인 뷰어 (직접 개발)
- `load_ply()` — 표준 3DGS PLY 로더. SH degree 자동 감지.
- `load_colmap_cameras()` — pycolmap으로 COLMAP sparse 로딩
- `load_cameras_json()` — cameras.json 로딩 (폴백)
- `compute_alignment()` — 두 모델의 좌표계 정렬 변환(similarity transform) 계산 (Procrustes 초기값)
- `_rasterize_rgb()` — 그래디언트가 흐르는 렌더 헬퍼 (photometric refinement용)
- `refine_alignment_photometric()` — Procrustes `T` 위에 Sim(3) residual을 L1+SSIM 손실로 최적화. 3가지 속도 최적화 내장 (아래 참고)
- `setup_initial_camera()` — 첫 번째 학습 카메라 시점으로 viser 초기화
- `render_scene()` — gsplat rasterization 래퍼. `transform` 인자로 c2w 좌표 변환 지원
- `viewer_render_fn()` — 단일/듀얼 분기. 듀얼 시 절반 너비로 렌더 후 hstack

### `gsplat/` — 핵심 라이브러리 (원본 gsplat, 수정 안 함)
- `rendering.py` — `rasterization()` 함수. Gaussian → 2D 이미지 래스터라이제이션
- `cuda/csrc/` — CUDA 커널 (projection, rasterize, SH)
- `cuda/_wrapper.py` — CUDA Python wrapper
- `strategy/` — Gaussian 밀도 제어 (DefaultStrategy, MCMCStrategy)
- `compression/` — 모델 압축 (PNG 기반)

### `examples/` — 원본 gsplat 예제 (수정 안 함)
- `simple_trainer.py` — 학습 스크립트 (.pt 체크포인트 출력)
- `simple_viewer.py` — 원본 뷰어 (.pt 전용, 카메라 초기화 없음)
- `datasets/colmap.py` — COLMAP 파서 (이미지 파일 필요)
- `datasets/normalize.py` — 좌표 정규화 (similarity_from_cameras + PCA)

### 기타
- `tests/` — 단위 테스트
- `docs/` — 문서
- `profiling/` — 성능 프로파일링

## 핵심 기술 사항

### 뷰어 아키텍처
viser(웹 3D 서버) + nerfview(렌더링 뷰어) 기반.
서버가 GPU에서 래스터라이제이션 수행 → 결과 이미지를 브라우저로 스트리밍.
`nerfview.Viewer`에 `render_fn` 콜백을 등록하면, 카메라 이동 시 자동으로 호출된다.

### 듀얼 뷰 구현 방식
두 씬을 각각 절반 너비(W/2)로 렌더링 → `np.hstack`으로 합쳐서 하나의 이미지로 반환.
카메라 상태가 하나이므로 동기화는 자동 보장. 별도 서버/iframe/동기화 로직 불필요.

### 듀얼 뷰 좌표계 정렬
두 모델은 서로 다른 COLMAP 결과/정규화로 학습될 수 있어 world 좌표계가 다르다.
SIBR 뷰어는 모델별로 자체 cameras.json을 사용해 독립적으로 렌더링하지만, 본 뷰어는
하나의 viser 카메라 상태를 공유하므로 좌표 변환이 필요하다.

해결 (2단계):

**1단계 — Procrustes 초기값**: 두 모델의 cameras.json에서 대응 카메라 위치쌍으로 Procrustes
분석(SVD)을 수행하여 모델A→모델B의 similarity transform `T`(scale + rotation + translation,
4x4)를 계산한다. 단, 이 단계는 카메라 **위치만** 사용하고 회전은 목적함수에 포함하지 않아
잔여 미스매치가 발생할 수 있다. `compute_alignment()` 참고.

**2단계 — Photometric refinement** (`--refine`, 기본 활성화): Procrustes 결과 `T_init` 위에
Sim(3) residual `ΔT`(10 DoF: 3D 평행이동 + 6D 회전 + 1D 로그 스케일)를 얹어,
실제 렌더 픽셀값의 L1 + SSIM 손실을 최소화하여 `T = T_init @ ΔT`를 구한다.
- gsplat `rasterization()`의 `viewmats` 미분 가능성을 활용하여 그래디언트 역전파.
- A 모델 렌더는 `torch.no_grad()`로 프리캐시하여 타겟 고정, B 모델 렌더만 ΔT를 통해 그래디언트 경로.
- 학습 카메라 전체에서 랜덤 샘플링 SGD (Adam, cosine decay), 기본 2000 이터레이션.
- 6D 회전 표현은 `examples/utils.py`의 `rotation_6d_to_matrix` 재사용.
- 뷰어 시작 시 오프라인으로 한 번만 실행. 런타임 렌더 경로는 변경 없음.
- `refine_alignment_photometric()` 참고.

**속도 최적화 (3가지, 기본 모두 활성화)**:
1. **모델 A 프리캐시**: 모델 A 렌더는 카메라별로 결과가 불변이므로, 루프 진입 전 N개 카메라를
   한 번만 렌더링하여 캐시. 루프 내에서는 텐서 조회만 수행. ~50% 렌더 절감.
2. **Coarse-to-fine 점진적 해상도** (`--refine_no_ctf`로 비활성화): `downscale < 8`이면
   자동으로 여러 stage로 분할 (예: downscale=1 → [8, 4, 2, 1]). 초반 stage는 작은 해상도로
   빠르게 coarse alignment, 마지막 stage에서 원본 해상도로 미세 조정. 각 stage마다 캐시 재생성.
   Optimizer/scheduler는 전체 iters에 걸쳐 공유.
3. **Early stopping** (`--refine_patience`, 기본 200, 0=비활성화): EMA loss가 patience step
   동안 threshold(1e-5) 이상 개선되지 않으면 해당 stage 조기 종료. Stage 전환 시 카운터 리셋.

최종 결과:
- 모델 A: viser c2w를 그대로 사용
- 모델 B: `c2w_B = T @ c2w_A`로 변환 후 렌더

이로써 viser 카메라는 모델 A의 좌표계에서 동작하고, 두 화면이 항상 같은 시점/같은 조작에
동기화된다. 정렬은 `cameras.json`이 양쪽 모두 있을 때만 활성화.

### PLY ↔ gsplat 텐서 변환
PLY와 gsplat의 활성화 공간은 동일:
- position: 직접 사용
- opacity: logit 공간 → `torch.sigmoid()` 적용
- scale: log 공간 → `torch.exp()` 적용
- quaternion (wxyz): 직접 사용 (gsplat이 내부에서 normalize)
- SH 계수: 직접 사용. PLY는 채널 우선 저장 → `reshape(N, 3, K).transpose(0, 2, 1)` 변환 필요

### 초기 카메라 설정
SIBR처럼 첫 번째 학습 카메라에서 시작:
- `c2w[:3, 3]` → position
- `c2w[:3, 2]` → forward (OpenCV +Z)
- `-c2w[:3, 1]` → up (OpenCV -Y)
- 전체 카메라의 평균 up → `server.scene.set_up_direction()`
- FOV: `2 * atan(height / (2 * fy))`
- viser API: `server.initial_camera.position/look_at/up/fov`

### 좌표계
- COLMAP/OpenCV: X-right, Y-down, Z-forward
- viser 기본: +Z up (Blender)
- `set_up_direction()`으로 씬의 실제 up 방향에 맞춤

## 의존성
viser (v1.0.26), nerfview (v0.1.2), pycolmap, torch, numpy, gsplat (로컬), fused-ssim (photometric refinement용)

## 데이터 위치
- gsplat 학습 데이터: `examples/data/360_v2/`
- gsplat 학습 결과: `examples/results/benchmark/`
- 테스트용 PLY: `/home/imlab/다운로드/bartender_srgs/`, `/home/imlab/다운로드/bartender_mrsr/`
