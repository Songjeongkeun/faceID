import cv2

_trackbar_created = False

def empty_callback(val):
    pass

def apply_sobel(frame, window_name="Control"):
    global _trackbar_created
    
    # 1. 필터가 처음 실행될 때 트랙바 생성
    if not _trackbar_created:
        cv2.createTrackbar("Sobel Scale", window_name, 1, 10, empty_callback)
        cv2.createTrackbar("Sobel Delta", window_name, 0, 255, empty_callback)
        _trackbar_created = True

    # 2. 트랙바 값 읽어오기
    scale = cv2.getTrackbarPos("Sobel Scale", window_name)
    delta = cv2.getTrackbarPos("Sobel Delta", window_name)

    if scale == 0:
        scale = 1
        cv2.setTrackbarPos("Sobel Scale", window_name, 1)

    # 3. 필터 적용 로직
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_frame, cv2.CV_16S, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray_frame, cv2.CV_16S, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)