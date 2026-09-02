# 트랙바X
# import cv2
# import sys

# def main():
#     # Load the image
#     src = cv2.imread("./images/park.png", cv2.IMREAD_GRAYSCALE)
    
#     scale = 1
#     delta = 0

#     if src is None:
#         print("Could not open or find the image!")
#         sys.exit(-1)

#     # X방향 미분 (Sobel)
#     grad_x = cv2.Sobel(src, cv2.CV_16S, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#     # Y방향 미분 (Sobel)
#     grad_y = cv2.Sobel(src, cv2.CV_16S, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

#     # 절대값 변환 및 8비트(uint8)로 스케일링
#     abs_grad_x = cv2.convertScaleAbs(grad_x)
#     abs_grad_y = cv2.convertScaleAbs(grad_y)
    
#     # X, Y 그래디언트 합성
#     grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
#     # 결과 출력
#     cv2.imshow("Image", src)
#     cv2.imshow("Sobel", grad)
    
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


# 트랙바
import cv2
import sys

# 콜백 함수에서 원본 이미지에 접근할 수 있도록 전역 변수로 선언
src = None

def update_sobel(val):
    """
    트랙바가 움직일 때마다 호출되는 콜백 함수
    """
    global src
    if src is None:
        return

    # 트랙바의 현재 값 읽어오기
    scale = cv2.getTrackbarPos("Scale", "Sobel")
    delta = cv2.getTrackbarPos("Delta", "Sobel")

    # scale이 0이면 결과가 검은색으로만 나오므로 최소값을 1로 보정
    if scale == 0:
        scale = 1
        cv2.setTrackbarPos("Scale", "Sobel", 1)

    # X, Y 방향 미분 (트랙바에서 읽어온 scale, delta 적용)
    grad_x = cv2.Sobel(src, cv2.CV_16S, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src, cv2.CV_16S, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    # 절대값 변환 및 8비트로 스케일링
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # 합성
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # 갱신된 결과 출력
    cv2.imshow("Sobel", grad)


def main():
    global src
    
    # 이미지 로드
    src = cv2.imread("./images/park.png", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Could not open or find the image!")
        sys.exit(-1)

    # 트랙바를 부착하기 위해 먼저 창을 생성해야 합니다.
    cv2.namedWindow("Sobel")

    # 트랙바 생성
    # 파라미터: 트랙바 이름, 부착할 창 이름, 초기값, 최대값, 콜백 함수
    cv2.createTrackbar("Scale", "Sobel", 1, 10, update_sobel)
    cv2.createTrackbar("Delta", "Sobel", 0, 255, update_sobel)

    # 원본 이미지 출력
    cv2.imshow("Image", src)
    
    # 초기 상태 출력을 위해 콜백 함수 1회 수동 호출
    update_sobel(0)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()