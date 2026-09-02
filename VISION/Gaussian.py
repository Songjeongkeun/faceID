import cv2
import sys

# 트랙바 콜백 함수
def on_trackbar(val):
    # 가우시안 블러의 커널 크기는 반드시 1 이상의 홀수여야 함
    ksize = val
    
    # 값이 0이면 1로 고정
    if ksize < 1:
        ksize = 1
    # 짝수라면 1을 더해 홀수로 만들어줌
    elif ksize % 2 == 0:
        ksize += 1
        
    # 가우시안 블러 적용
    dst = cv2.GaussianBlur(src, (ksize, ksize), 0)
    
    # 결과 출력
    cv2.imshow("Gaussian Blur", dst)

# 메인 실행부
if __name__ == "__main__":
    # 이미지 읽기
    src = cv2.imread("./images/park.png")

    # 예외 처리
    if src is None:
        print("이미지를 찾을 수 없습니다. 경로를 확인해 주세요.")
        sys.exit()

    # 트랙바를 붙일 창을 먼저 생성해야 함
    cv2.namedWindow("Gaussian Blur")

    # 트랙바 생성
    # 파라미터: 트랙바 이름, 창 이름, 초기값, 최대값(60), 콜백 함수
    cv2.createTrackbar("Kernel Size", "Gaussian Blur", 1, 60, on_trackbar)

    # 프로그램을 처음 실행했을 때 초기 화면을 보여주기 위해 강제 호출
    # on_trackbar(1)

    # 원본 이미지 창 (비교용)
    cv2.imshow("src", src)

    # 키 입력 대기 (아무 키나 누르면 종료)
    cv2.waitKey(0)
    
    # 모든 창 닫기
    cv2.destroyAllWindows()