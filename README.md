# Facial Landmark Detection using Stacked Hourglass Networks

Stacked Hourglass Network를 이용한 Facial landmark detection model입니다. Stacked Hourglass Network는 본래 Human Pose Estimation을 위해 고안되었으나, 얼굴 특징점 검출 등 타 분야에서도 다양하게 사용되고 있습니다.

```bash
├── README.md
├── data
│   └── dataset_parser.ipynb            # 1. 300W Dataset을 구성하기 위한 코드입니다. 
├── data_augmentation_examples.ipynb    # 2. Dataset augmentation의 결과를 시각화하였습니다.
├── evaluate.py                         # 3. Evaluation 코드입니다.
├── inference.py                        # 4. Inference 코드입니다.
├── libs
│   ├── cascade_classifier.py           # cascade 방식의 face detector
│   ├── dp.py                           # Data loader for Model training
│   ├── eval.py                         # Evaluation을 위한 기능
│   └── utils.py     
├── requirements.txt
├── tasks
│   ├── model_SHG.py                    # Stacked Hourglass Network 모델 구현
│   ├── model_SHG_TC.py                 # Transposed Convolutional Layer 적용(experimental)
│   └── model_UNet.py                   # UNet
└── train_fit.py                        # 학습 코드
```

## Requirements

    tensorflow >= 2.6.0

## To be implemented
    
    - BlazeFace for Face detector