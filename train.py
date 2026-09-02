import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "vggface_classifier.pth")

IMAGE_SIZE = (160, 160)  # VGGFace 표준 입력 크기
BATCH_SIZE = 8
EPOCHS = 5 
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print(" 딥러닝 얼굴 인식 전이학습을 시작합니다.")

    # 1. 데이터셋 확인
    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print("[오류] 학습할 데이터셋 폴더가 없습니다. 웹캠에서 먼저 얼굴을 수집하세요.")
        return

    # 2. 전처리(Transform) 및 데이터로더 설정
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 폴더 이름을 기준으로 클래스(사람 이름) 자동 분류
    dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    print(f"총 {num_classes}명의 데이터를 학습합니다: {class_names}")
    print(f"총 이미지 수: {len(dataset)}장\n")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. VGGFace(InceptionResnetV1) 사전 학습 모델 로드
    print("VGGFace 사전 학습 가중치를 불러오는 중...")
    model = InceptionResnetV1(pretrained='vggface2', classify=True).to(DEVICE)

    # 기존 특징 추출기 레이어들을 동결(Freeze)하여 가중치 파괴 방지
    for param in model.parameters():
        param.requires_grad = False

    # 맨 마지막 분류기 레이어를 우리가 등록한 사람 수에 맞게 교체
    in_features = model.logits.in_features
    model.logits = nn.Linear(in_features, num_classes).to(DEVICE)

    # 4. 손실함수 및 옵티마이저 설정 (새로 교체한 레이어만 학습)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.logits.parameters(), lr=LEARNING_RATE)

    # 5. 모델 학습
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_preds = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # 기울기 초기화 -> 순전파 -> 손실 계산 -> 역전파 -> 가중치 업데이트
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct_preds.double() / len(dataset)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    # 6. 학습 완료된 모델 가중치 및 클래스 이름 저장
    torch.save({
        'model_state': model.state_dict(),
        'classes': class_names
    }, MODEL_SAVE_PATH)
    
    print(f"\n[성공] 학습이 완료되었습니다!")
    print(f"[저장 위치] {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()