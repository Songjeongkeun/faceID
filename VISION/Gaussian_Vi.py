import cv2
import sys

def empty_callback(val):
    """
    동영상 재생 루프 안에서 트랙바 값을 직접 읽어올 것이므로,
    콜백 함수는 비워둡니다.
    """
    pass

def main():
    # 1. 카메라 또는 동영상 파일 열기 (웹캠은 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("동영상이나 카메라를 열 수 없습니다. 경로를 확인해 주세요.")
        sys.exit(-1)

    # 창 생성
    cv2.namedWindow("Gaussian Blur Video")

    # 2. 트랙바 생성
    # 파라미터: 트랙바 이름, 창 이름, 초기값(1), 최대값(60), 콜백 함수
    cv2.createTrackbar("Kernel Size", "Gaussian Blur Video", 1, 60, empty_callback)

    print("동영상이 재생 중입니다. 종료하려면 'q' 키를 누르세요.")

    # 3. 프레임 반복 처리 (동영상 재생)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("동영상이 끝났거나 프레임을 읽을 수 없습니다.")
            break
            
        # 트랙바의 현재 값(커널 크기) 읽어오기
        ksize = cv2.getTrackbarPos("Kernel Size", "Gaussian Blur Video")
        
        # 4. 가우시안 블러의 커널 크기 보정 (반드시 1 이상의 홀수여야 함)[cite: 4]
        if ksize < 1:
            ksize = 1 # 값이 0이면 1로 고정[cite: 4]
        elif ksize % 2 == 0:
            ksize += 1 # 짝수라면 1을 더해 홀수로 만들어줌[cite: 4]
            
        # 가우시안 블러 적용[cite: 4]
        dst = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        
        # 원본 컬러 영상과 블러 처리된 영상 출력
        cv2.imshow("Original Video", frame)
        cv2.imshow("Gaussian Blur Video", dst)
        
        # 5. 'q' 키를 누르면 종료 (30ms 대기)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 자원 해제 및 모든 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()