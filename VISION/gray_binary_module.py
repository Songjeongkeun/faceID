import cv2

_trackbar_created = False

def empty_callback(val):
    pass

def apply_binary(frame, window_name="Control"):
    global _trackbar_created
    
    # 1. 트랙바 최초 1회 생성 (초기값 128, 최대값 255)
    if not _trackbar_created:
        cv2.createTrackbar("Binary Threshold", window_name, 128, 255, empty_callback)
        _trackbar_created = True

    # 2. 트랙바 값 읽어오기
    current_threshold = cv2.getTrackbarPos("Binary Threshold", window_name)
    
    # 3. 흑백 변환 및 이진화 적용
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, dst = cv2.threshold(gray_frame, current_threshold, 255, cv2.THRESH_BINARY)
    
    # FaceID.py의 화면 출력 형식을 맞추기 위해 다시 3채널(BGR)로 변환
    return cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)