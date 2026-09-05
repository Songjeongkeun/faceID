import cv2

_trackbar_created = False
max_lowThreshold = 100
ratio = 3
kernel_size = 3

def empty_callback(val): # OpenCV 함수는 구조상 콜백 함수를 무조건 하나 내놓으라고 요구하기 때문(python OpenCV에선 그럼)
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
    
    # 4. Canny 엣지 검출          이미지           최소 임계값      최대 임계값             커널(소벨) 크기
    detected_edges = cv2.Canny(blurred_gray, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    
    # 5. 원본 컬러 프레임에 엣지 마스크 적용
    #      비트 AND 연산   컬러    컬러     mask 옵션으로 원본 영상에서 엣지가 있는 부분만 색깔을 남기고, 나머지는 싹 다 까맣게 지움
    dst = cv2.bitwise_and(frame, frame, mask=detected_edges)
    
    return dst # detected_edges # dst 지우면 컬러 아님