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
    cv2.namedWindow("Binary Video")

    # 2. 트랙바 생성
    # 초기값 128, 최대값 255 설정
    cv2.createTrackbar("Threshold", "Binary Video", 128, 255, empty_callback)

    print("동영상이 재생 중입니다. 종료하려면 'q' 키를 누르세요.")

    # 3. 프레임 반복 처리 (동영상 재생)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("동영상이 끝났거나 프레임을 읽을 수 없습니다.")
            break

        # 이진화(Threshold)를 위해 현재 컬러 프레임을 흑백(Grayscale)으로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 트랙바의 현재 값(임계값) 읽어오기
        current_threshold = cv2.getTrackbarPos("Threshold", "Binary Video")
        
        # 4. 이진화 적용
        # 현재 트랙바 값보다 크면 255(흰색), 작으면 0(검은색)으로 변환
        _, dst = cv2.threshold(gray_frame, current_threshold, 255, cv2.THRESH_BINARY)
        
        # 원본 컬러 영상과 이진화된 영상 출력
        cv2.imshow("Original Video", frame)
        cv2.imshow("Binary Video", dst)
        
        # 5. 'q' 키를 누르면 종료 (30ms 대기)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 자원 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()