import cv2
import sys

# 전역 변수 설정 (기존 코드와 동일)
max_lowThreshold = 100
ratio = 3
kernel_size = 3

def empty_callback(val):
    """
    동영상 재생 루프 안에서 트랙바 값을 직접 읽어올 것이므로,
    콜백 함수는 비워둡니다.
    """
    pass

def main():
    # 1. 동영상 파일 또는 웹캠 열기
    # 웹캠은 0, 동영상 파일은 "video.mp4" 등 경로 입력
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("동영상이나 카메라를 열 수 없습니다. 경로를 확인해 주세요.")
        sys.exit(-1)

    # 2. 창 생성 및 창 크기 조절 허용
    cv2.namedWindow("Canny Video", cv2.WINDOW_NORMAL)
    
    # 3. 트랙바 생성
    cv2.createTrackbar("Threshold", "Canny Video", 0, max_lowThreshold, empty_callback)

    print("동영상이 재생 중입니다. 종료하려면 'q' 키를 누르세요.")

    # 4. 프레임 반복 처리 (동영상 재생)
    while True:
        # 프레임 읽기 (컬러 이미지로 읽어옴)
        ret, frame = cap.read()
        
        if not ret:
            print("동영상이 끝났거나 프레임을 읽을 수 없습니다.")
            break
            
        # 트랙바의 현재 Threshold 값 읽어오기
        lowThreshold = cv2.getTrackbarPos("Threshold", "Canny Video")

        # Canny 알고리즘 연산을 위해 현재 프레임을 흑백으로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거 (블러 처리)
        blurred_gray = cv2.blur(gray_frame, (3, 3))
        
        # Canny 엣지 검출 연산
        detected_edges = cv2.Canny(blurred_gray, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
        
        # 원본 컬러 프레임에 엣지 마스크 적용 (엣지 부분만 원본 색상으로 출력됨)
        dst = cv2.bitwise_and(frame, frame, mask=detected_edges)
        
        # 화면 출력
        cv2.imshow("Original Video", frame)
        cv2.imshow("Canny Video", dst)
        
        # 5. 종료 조건: 30ms 대기하며 'q' 키 입력 확인
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 자원 해제 및 모든 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()