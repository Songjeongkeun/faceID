import cv2

_trackbar_created = False

def empty_callback(val):
    pass

def apply_gaussian(frame, window_name="Control"):
    global _trackbar_created
    
    # 처음 실행될 때 트랙바 생성
    if not _trackbar_created:
        cv2.createTrackbar("Gaussian Kernel", window_name, 1, 60, empty_callback)
        _trackbar_created = True

    # 트랙바 값 읽어오기
    ksize = cv2.getTrackbarPos("Gaussian Kernel", window_name)
    
    if ksize < 1:
        ksize = 1
    elif ksize % 2 == 0:
        ksize += 1
        
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)