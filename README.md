# VLM LoRA Fine-tuning

비전-언어 모델(VLM)에 대한 LoRA(Low-Rank Adaptation) 미세 조정을 위한 파이썬 프레임워크입니다.

## 개요

이 프로젝트는 Qwen 및 Gemma와 같은 다양한 비전-언어 모델(VLM)에 LoRA 미세 조정을 적용하기 위한 도구를 제공합니다. ChartQA 및 RealWorldQA와 같은 데이터셋을 사용하여 다양한 시각적 추론 작업에 대해 모델을 미세 조정할 수 있습니다.

## 지원 모델

- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct
- Qwen/Qwen2.5-VL-72B-Instruct
- google/gemma-3-4b-pt
- google/gemma-3-12b-pt
- google/gemma-3-27b-pt

## 지원 데이터셋

- HuggingFaceM4/ChartQA
- xai-org/RealworldQA
- Lmms-lab/RealWorldQA

## 설치

필요한 라이브러리를 설치하려면:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 학습 스크립트 실행

```bash
./train.sh
```

또는 사용자 지정 매개변수로 직접 실행:

```bash
accelerate launch train.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "HuggingFaceM4/ChartQA" \
    --num_train_epochs 3 \
    --output_dir "results"
```

## 프로젝트 구조

```
.
├── train.py            # 주요 학습 스크립트
├── train.sh            # 학습을 위한 편의성 쉘 스크립트
├── requirements.txt    # 필요한 패키지 목록
├── model/              # 모델 관련 코드 디렉토리
│   ├── __init__.py     # 모델 매핑 정의
│   ├── qwen_qwen2_5_vl.py    # Qwen 모델 구현
│   ├── google_gemma_3.py     # Gemma 모델 구현
│   └── ...
├── dataset/            # 데이터셋 처리 코드 디렉토리
│   ├── __init__.py     # 데이터셋 매핑 정의
│   ├── HuggingFaceM4___chart_qa.py    # ChartQA 데이터셋 처리
│   └── ...
└── tools/              # 유틸리티 및 도구 디렉토리
```

## 주요 기능

- **LoRA 미세 조정**: 메모리 효율적인 LoRA를 사용하여 대규모 VLM 모델을 미세 조정합니다.
- **4비트 양자화**: 효율적인 학습을 위한 BitsAndBytes 라이브러리를 사용한 4비트 양자화 지원
- **다양한 모델 지원**: Qwen 및 Gemma 모델 제품군 지원
- **다양한 데이터셋**: 시각적 질문 응답 작업을 위한 다양한 데이터셋 지원

## 참고 사항

- 이 프로젝트는 비전-언어 모델의 효율적인 미세 조정을 위해 설계되었습니다.
- 학습 시 텐서보드 또는 Weights & Biases를 통한 진행 상황 모니터링이 가능합니다.
- 모든 모델은 HuggingFace에서 로드됩니다.
