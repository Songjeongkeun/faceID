import cv2

_trackbar_created = False

def empty_callback(val):    # OpenCV 함수는 구조상 콜백 함수를 무조건 하나 내놓으라고 요구하기 때문(python OpenCV에선 그럼)
    pass

def apply_binary(frame, window_name="Control"):
    global _trackbar_created
    
    # 1. 트랙바 최초 1회 생성 (초기값 128, 최대값 255)
    if not _trackbar_created:
        cv2.createTrackbar("Binary Threshold", window_name, 128, 255, empty_callback)
        _trackbar_created = True

    # 2. 흑백 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. 트랙바 값 읽어오기           현재 트랙바 값보다 크면 255(흰) 같거나 작으면 0(검)
    current_threshold = cv2.getTrackbarPos("Binary Threshold", window_name)
    
    # 4. 이진화 적용 (현재 값보다 크면 흰색, 작으면 검은색)
    _, dst = cv2.threshold(gray_frame, current_threshold, 255, cv2.THRESH_BINARY)
    
    return dst