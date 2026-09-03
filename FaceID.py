import cv2
import os
import time
import numpy as np
import torch
import torch.nn as nn
from VISION import gray_binary_module
from VISION import canny_module
from VISION import sobel_module
from VISION import gaussian_module
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
# 라이브러리 설치 ↓
# pip install facenet-pytorch torch torchvision opencv-python 

# 모델 학습 : models/InceptionResnetV1_train.py 실행


# 실행 위치와 관계없이 이 파일 옆의 dataset 폴더에 저장
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "vggface_classifier.pth")

# 웹캠 번호: 기본 웹캠은 보통 0
CAMERA_INDEX = 0

# 얼굴 이미지를 저장할 크기
SAVE_SIZE = (224, 224)

# 사람 한 명당 수집할 얼굴 이미지 수
TARGET_IMAGE_COUNT = 100

# 이미지 저장 간격(초)
SAVE_INTERVAL = 0.2

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# 추론용 전처리 Transform
transform_infer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_trained_model():
    """학습된 InceptionResnetV1 모델(.pth)을 불러옵니다."""
    if not os.path.exists(MODEL_PATH):
        print("모델 파일이 없습니다.")
        return None, None

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        class_names = checkpoint['classes']

        # Pretrained 백본 생성 후 분류기 교체
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=len(class_names))        

        model.load_state_dict(checkpoint['model_state'])
        model.to(DEVICE)
        model.eval()
        print(f"모델 로드 성공! 등록된 클래스: {class_names}")
        return model, class_names
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return None, None

def predict_identity(model, class_names, face_img):
    """자른 얼굴 영역(ROI)을 입력받아 사람 이름과 확률을 반환합니다."""
    if face_img is None or face_img.size == 0 or model is None:
        return "", 0.0

    # BGR -> RGB 변환
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    input_tensor = transform_infer(rgb_face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)

    name = class_names[top_class.item()]
    confidence = top_prob.item()
    return name, confidence

def get_largest_face(faces):
    """
    웹캠에 여러 명이 잡혔을 때,
    가장 크게 검출된 얼굴 하나를 선택하는 함수
    """
    if len(faces) == 0:
        return None

    # 얼굴 박스의 넓이(w * h)가 가장 큰 얼굴 반환
    return max(faces, key=lambda face: face[2] * face[3])


def apply_filter(frame, mode, window_name="Smart Camera"):
    """
    선택된 필터를 웹캠 프레임에 적용하는 함수

    mode
    0: 원본
    1: 흑백
    2: 캐니
    3: 소벨
    4: 가우시안 블러
    """

    if mode == 0:
        return frame

    if mode == 1:
        # 1. Grayscale & Binary 필터 호출
        return gray_binary_module.apply_binary(frame, window_name)
    elif mode == 2:
        # 2. Canny 필터 호출
        return canny_module.apply_canny(frame, window_name)
    elif mode == 3:
        # 3. Gaussian Blur 필터 호출
        return gaussian_module.apply_gaussian(frame, window_name)
    elif mode == 4:
        # 4. Sobel 필터 호출
        return sobel_module.apply_sobel(frame, window_name)
        

    return frame


def draw_glasses(frame, x, y, w, h):
    """AR 아이템 1: 얼굴 위치를 기준으로 안경 그리기"""

    # 얼굴 박스 높이를 기준으로 눈 위치를 대략 계산
    eye_y = y + int(h * 0.40)

    left_center = (x + int(w * 0.30), eye_y)
    right_center = (x + int(w * 0.70), eye_y)

    radius = int(w * 0.18)
    thickness = max(2, int(w * 0.03))

    # 안경 렌즈 2개
    cv2.circle(frame, left_center, radius, (0, 0, 0), thickness)
    cv2.circle(frame, right_center, radius, (0, 0, 0), thickness)

    # 렌즈 사이 연결 부분
    cv2.line(
        frame,
        (left_center[0] + radius, eye_y),
        (right_center[0] - radius, eye_y),
        (0, 0, 0),
        thickness,
    )


def draw_hat(frame, x, y, w, h):
    """AR 아이템 2: 얼굴 위에 모자 그리기"""

    # 얼굴 바깥 위쪽에도 모자가 보이도록 y값을 조절
    hat_top = max(0, y - int(h * 0.28))
    hat_bottom = y + int(h * 0.08)

    # 모자 몸통
    cv2.rectangle(
        frame,
        (x + int(w * 0.15), hat_top),
        (x + int(w * 0.85), hat_bottom),
        (255, 80, 30),
        -1,
    )

    # 모자 챙
    cv2.ellipse(
        frame,
        (x + w // 2, hat_bottom),
        (int(w * 0.55), int(h * 0.10)),
        0,
        0,
        180,
        (255, 80, 30),
        -1,
    )


def draw_mustache(frame, x, y, w, h):
    """AR 아이템 3: 얼굴 아래쪽에 콧수염 그리기"""

    center_x = x + w // 2
    center_y = y + int(h * 0.68)
    size = int(w * 0.18)

    # 타원 2개를 이용해 콧수염 모양 생성
    cv2.ellipse(
        frame,
        (center_x - size // 2, center_y),
        (size, size // 2),
        20,
        0,
        180,
        (40, 20, 10),
        -1,
    )

    cv2.ellipse(
        frame,
        (center_x + size // 2, center_y),
        (size, size // 2),
        160,
        0,
        180,
        (40, 20, 10),
        -1,
    )


def register_face(frame, face, user_name, image_count):
    """
    검출된 얼굴 영역만 잘라서 dataset/이름 폴더에 저장
    """

    x, y, w, h = face

    # 원본 프레임에서 얼굴 영역만 자르기
    face_img = frame[y:y + h, x:x + w]

    # 얼굴 이미지가 비어 있으면 저장하지 않음
    if face_img.size == 0:
        return False

    # 모델 입력 크기에 맞게 224 x 224로 변경
    face_img = cv2.resize(face_img, SAVE_SIZE)

    # 예: dataset/song/ 폴더 생성
    save_dir = os.path.join(DATASET_DIR, user_name)
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f"{user_name}_{image_count:03d}.jpg")
    cv2.imwrite(file_path, face_img)
    return True



def main():
    # 웹캠 열기
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 웹캠 출력 크기 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # OpenCV에 포함된 Haar Cascade 얼굴 검출기 불러오기
    cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )
    face_cascade = cv2.CascadeClassifier(cascade_path)
    # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


    if face_cascade.empty():
        print("얼굴 검출기를 불러오지 못했습니다.")
        return

    # 화면 창 생성
    cv2.namedWindow("Smart Camera")

    filter_mode = 0
    ar_item = 0

    register_mode = False
    user_name = ""
    image_count = 0
    last_save_time = 0

    # InceptionResnetV1 모델 로드
    model, class_names = load_trained_model()

    while True:
        # 웹캠 프레임 읽기
        success, frame = cap.read()

        if not success:
            print("웹캠 프레임을 읽지 못했습니다.")
            break

        # 거울처럼 보이도록 좌우 반전
        frame = cv2.flip(frame, 1)

        # 얼굴 등록용 원본 이미지 보관
        original_frame = frame.copy()

        # Haar Cascade 얼굴 검출을 위해 흑백 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 위치 검출
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        # 여러 얼굴 중 가장 큰 얼굴 한 명만 선택
        face = get_largest_face(faces)

        if face is not None:
            x, y, w, h = face

            # 검출된 얼굴 영역 표시
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            # 선택된 AR 아이템 적용
            if ar_item == 1:
                draw_glasses(frame, x, y, w, h)

            elif ar_item == 2:
                draw_hat(frame, x, y, w, h)

            elif ar_item == 3:
                draw_mustache(frame, x, y, w, h)

            # ------------------------------
            # 실시간 얼굴 인식 및 결과 표시 (InceptionResnetV1)
            # ------------------------------
            if model is not None and not register_mode:
                face_img = original_frame[y:y+h, x:x+w]
                name, confidence = predict_identity(model, class_names, face_img)

# 방법 1 --------------------------------------------- unknown---------------
                # # 확률이 85% 미만이면 Unknown 처리
                # THRESHOLD = 0.85
                # if confidence < THRESHOLD:
                #     display_name = "Unknown"
                #     color = (0, 0, 255) # red
                # else:
                #     display_name = name
                #     color = (0, 255, 0) # green

                # # 얼굴 박스 상단에 결과 표시
                # label_text = f"{name} ({confidence * 100:.1f}%)"

                # # putText 좌표 int 변환 및 상단 잘림 방지
                # text_x = int(x)
                # text_y = int(max(30, y - 10))
                # cv2.putText(
                #     frame, 
                #     label_text, 
                #     (text_x, text_y), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.7, 
                #     (0, 0, 0), 
                #     4
                # )
                # cv2.putText(
                #     frame,
                #     label_text,
                #     (text_x, text_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 255, 0),
                #     2,
                # )
# ------------------------------------------------

# 방법2-------------------------------------------
                # 확률이 85% 미만이면 Unknown 처리
                # THRESHOLD = 0.85
                # color = (0, 255, 0)

                # 방법 1정리

                display_name = name
  
                label_text = f"{display_name} ({confidence * 100:.1f}%)"
                
                    
                cv2.putText(
                    frame,
                    label_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0 , 255, 0),  # color
                    2,
                )


# --------------------------------------------------------
# 방법 3 -------------------------------------------- 네모 칸만 뜸
            # 확률이 85% 미만이면 Unknown 처리
            # THRESHOLD = 0.85
            # if confidence < THRESHOLD:
            #     display_name = "Unknown"
            #     color = (0, 255, 0) # 아니면 색만 변경 가능
            # else:
            #     display_name = name
            #     color = (0, 255, 0) # green

            # # 얼굴 박스 상단에 결과 표시
            # if display_name == "Unknown":
            #     label_text = ""
            # else:
            #     label_text = f"{display_name} ({confidence * 100:.1f}%)"
                
            # cv2.putText(
            #     frame,
            #     label_text,
            #     (x, y - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     color,
            #     2,
            #                 )
# -------------------------------------------------------
            # 얼굴 등록 모드일 때 일정 간격으로 이미지 저장
            now = time.time()

            if (
                register_mode
                and image_count < TARGET_IMAGE_COUNT
                and now - last_save_time >= SAVE_INTERVAL
            ):
                saved = register_face(
                    original_frame,
                    face,
                    user_name,
                    image_count,
                )

                # 파일 저장에 성공한 경우에만 개수 증가
                if saved:
                    image_count += 1
                    last_save_time = now

            # 목표 수만큼 저장하면 등록 종료
            if register_mode and image_count >= TARGET_IMAGE_COUNT:
                register_mode = False
                print(
                    f"{user_name} 등록 완료: "
                    f"{TARGET_IMAGE_COUNT}장"
                )
                # break

        # 현재 선택된 영상 필터 적용
        result = apply_filter(
            frame,
            filter_mode,
            "Smart Camera"
        )

        # 현재 필터와 AR 아이템 상태 표시
        cv2.putText(
            result,
            f"Filter: {filter_mode} | AR: {ar_item}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # 등록 중이면 저장 진행 상태 표시
        if register_mode:
            cv2.putText(
                result,
                f"Registering {user_name}: "
                f"{image_count}/{TARGET_IMAGE_COUNT}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            if face is None:
                cv2.putText(
                    result,
                    "No face detected",
                    (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # 결과 화면 출력
        cv2.imshow("Smart Camera", result)

        # 키보드 입력 읽기
        key = cv2.waitKey(1) & 0xFF

        # q: 프로그램 종료
        if key == ord("q"):
            break

        # 0~4: 필터 변경
        elif key in [ord("0"), ord("1"), ord("2"), ord("3"), ord("4")]:
            new_mode = key - ord("0")
            
            # 다른 번호의 필터를 눌러 모드가 변경되었을 때만 실행
            if new_mode != filter_mode:
                filter_mode = new_mode
                
                # 1. 창을 강제로 닫고 다시 열어서 기존 트랙바들을 전부 날림
                cv2.destroyWindow("Smart Camera")
                cv2.namedWindow("Smart Camera")
                
                # 2. 각 필터 모듈의 생성 플래그를 초기화해서 다시 그릴 수 있게 만듦
                gray_binary_module._trackbar_created = False
                canny_module._trackbar_created = False
                gaussian_module._trackbar_created = False
                sobel_module._trackbar_created = False
        

        # a: AR 아이템 순서대로 변경
        elif key == ord("a"):
            ar_item = (ar_item + 1) % 4

        # n: 등록할 사람 이름 입력
        elif key == ord("n"):
            user_name = input("등록할 이름 입력: ").strip()
            print(f"등록 준비 완료: {user_name} ('r' 키를 누르면 수집이 시작됩니다)")

        # r: 얼굴 등록 시작
        elif key == ord("r"):
            if user_name:
                register_mode = True
                image_count = 0
                last_save_time = 0
                print(f"{user_name} 얼굴 등록을 시작합니다.")
            else:
                print("'n' 키를 눌러 이름을 먼저 입력하세요.")

        # l: vggface_classifier.pth 로드
        elif key == ord("l"):
            model, class_names = load_trained_model()

    # 웹캠 연결 해제 및 모든 OpenCV 창 닫기
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
