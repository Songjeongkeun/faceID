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

    # 중심점이 필요하기 때문에 홀수로 맞춰야 함
    if ksize < 1:
        ksize = 1
    elif ksize % 2 == 0:
        ksize += 1

    # 커널이 커질수록 더 넓은 면적의 픽셀들을 섞어 더 강하게 흐려짐
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)