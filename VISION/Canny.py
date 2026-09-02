import cv2
import sys

# 전역 변수 설정
max_lowThreshold = 100
ratio = 3
kernel_size = 3

# 트랙바 콜백 함수
def CannyThreshold(val):
    lowThreshold = val
    
    detected_edges = cv2.blur(src, (3, 3))
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    dst = cv2.bitwise_and(src, src, mask=detected_edges)
    
    cv2.imshow("Image", src)
    cv2.imshow("Canny", dst)

# 메인 실행부
if __name__ == "__main__":
    src = cv2.imread("./images/park.png", cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        print("이미지를 찾을 수 없습니다. 경로를 확인해 주세요.")
        sys.exit(-1)

    # 1. WINDOW_NORMAL로 변경: 사용자가 창 크기를 조절할 수 있도록 허용
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
    
    # 2. 초기 창 크기 강제 설정: 트랙바가 잘리지 않도록 너비를 넉넉하게(최소 500px) 확보
    height, width = src.shape
    cv2.resizeWindow("Canny", max(width, 500), height + 50)
    
    # 3. 트랙바 이름 축소: 텍스트가 너무 길면 여전히 잘릴 수 있으므로 간결하게 변경
    cv2.createTrackbar("Threshold", "Canny", 0, max_lowThreshold, CannyThreshold)
    
    CannyThreshold(0)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# import cv2
# import sys

# # 전역 변수 설정
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 3

# # 트랙바 콜백 함수
# def CannyThreshold(val):
#     lowThreshold = val
    
#     # 1. 엣지 검출은 '흑백 이미지'를 사용
#     blurred_gray = cv2.blur(src_gray, (3, 3))
#     detected_edges = cv2.Canny(blurred_gray, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    
#     # 2. 마스크 연산은 '원본 컬러 이미지'에 적용
#     dst = cv2.bitwise_and(src_color, src_color, mask=detected_edges)
    
#     cv2.imshow("Image", src_color)
#     cv2.imshow("Canny", dst)

# # 메인 실행부
# if __name__ == "__main__":
#     # 1. 이미지를 컬러로 읽기 (cv2.IMREAD_COLOR 적용)
#     src_color = cv2.imread("./images/park.png", cv2.IMREAD_COLOR)
    
#     if src_color is None:
#         print("이미지를 찾을 수 없습니다. 경로를 확인해 주세요.")
#         sys.exit(-1)

#     # 2. Canny 알고리즘 연산을 위해 컬러 이미지를 흑백으로 변환한 복사본 생성
#     src_gray = cv2.cvtColor(src_color, cv2.COLOR_BGR2GRAY)

#     cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
    
#     # 컬러 이미지의 shape는 (height, width, channel)이므로 언패킹 주의
#     height, width, _ = src_color.shape
#     cv2.resizeWindow("Canny", max(width, 500), height + 50)
    
#     cv2.createTrackbar("Threshold", "Canny", 0, max_lowThreshold, CannyThreshold)
    
#     CannyThreshold(0)
    
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()