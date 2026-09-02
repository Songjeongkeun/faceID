import cv2

_trackbar_created = False
max_lowThreshold = 100
ratio = 3
kernel_size = 3

def empty_callback(val):
    pass

def apply_canny(frame, window_name="Control"):
    global _trackbar_created
    
    # 1. 트랙바 최초 1회 생성 (최대값 100)
    if not _trackbar_created:
        cv2.createTrackbar("Canny Threshold", window_name, 0, max_lowThreshold, empty_callback)
        _trackbar_created = True

    # 2. 트랙바 값 읽어오기
    lowThreshold = cv2.getTrackbarPos("Canny Threshold", window_name)
    
    # 3. 흑백 변환 및 블러 처리
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.blur(gray_frame, (3, 3))
    
    # 4. Canny 엣지 검출
    detected_edges = cv2.Canny(blurred_gray, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    
    # 5. 원본 컬러 프레임에 엣지 마스크 적용
    dst = cv2.bitwise_and(frame, frame, mask=detected_edges)
    
    return dst