import cv2
import sys

# 트랙바 콜백 함수
def on_trackbar(val):
    # 트랙바에서 넘겨받은 값(val)을 새로운 임계값으로 사용
    _, dst = cv2.threshold(src, val, 255, cv2.THRESH_BINARY)
    
    # 결과를 "Binary Image" 창에 갱신하여 출력
    cv2.imshow("Binary Image", dst)

# 메인 실행부
if __name__ == "__main__":
    # 흑백으로 이미지 읽기 (실제 이미지 경로로 변경해 주세요)
    src = cv2.imread("./images/park.png", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Could not open or find the image!")
        sys.exit()

    # 원본 이미지 출력
    cv2.imshow("Original Image", src)

    # 트랙바를 붙이기 위해 결과 창을 미리 생성
    cv2.namedWindow("Binary Image")

    # 트랙바 생성
    # 파라미터: 트랙바 이름, 붙일 창 이름, 초기값(128), 최대값(255), 콜백 함수
    # 이미지의 픽셀 값은 0~255 사이이므로 최대값을 255로 설정합니다.
    cv2.createTrackbar("Threshold", "Binary Image", 128, 255, on_trackbar)

    # 프로그램 실행 시 초기 화면을 보여주기 위해 강제 호출
    on_trackbar(128)

    # 키 입력 대기 및 창 닫기
    cv2.waitKey(0)
    cv2.destroyAllWindows()