import cv2
import sys

def empty_callback(val):
    """
    트랙바가 움직일 때 호출될 더미 콜백 함수.
    동영상 처리 루프 안에서 상태를 읽어올 것이므로 별도의 작업은 필요하지 않습니다.
    """
    pass

def main():
    # 1. 동영상 파일 또는 웹캠 열기
    # 웹캠을 사용하려면 0을 입력하고, 동영상 파일을 사용하려면 파일 경로(예: "./video.mp4")를 입력하세요.
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("동영상이나 카메라를 열 수 없습니다. 경로를 다시 확인해 주세요.")
        sys.exit(-1)

    # 트랙바를 부착하기 위해 먼저 창을 생성
    cv2.namedWindow("Sobel Video")

    # 2. 트랙바 생성
    cv2.createTrackbar("Scale", "Sobel Video", 1, 10, empty_callback)
    cv2.createTrackbar("Delta", "Sobel Video", 0, 255, empty_callback)

    print("동영상이 재생 중입니다. 종료하려면 'q' 키를 누르세요.")

    # 3. 프레임 반복 처리 (동영상 재생)
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        # 동영상이 끝나거나 프레임을 읽지 못하면 루프 종료
        if not ret:
            print("동영상이 끝났거나 프레임을 읽을 수 없습니다.")
            break

        # Sobel 필터는 주로 흑백 이미지에서 엣지를 검출하므로 컬러를 흑백으로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 트랙바의 현재 값 읽어오기
        scale = cv2.getTrackbarPos("Scale", "Sobel Video")
        delta = cv2.getTrackbarPos("Delta", "Sobel Video")

        # scale이 0이면 결과가 검은색으로만 나오므로 최소값을 1로 보정
        if scale == 0:
            scale = 1
            cv2.setTrackbarPos("Scale", "Sobel Video", 1)

        # 4. X, Y 방향 미분 (트랙바 값 적용)
        grad_x = cv2.Sobel(gray_frame, cv2.CV_16S, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_16S, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        # 절대값 변환 및 8비트로 스케일링
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # 합성
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # 원본과 필터가 적용된 결과 출력
        cv2.imshow("Original Video", frame)
        cv2.imshow("Sobel Video", grad)
        
        # 5. 종료 조건 (30ms 대기하며 'q' 키 입력을 확인)
        # 동영상 재생 속도를 조절하고 싶다면 30이라는 숫자를 변경하세요 (작을수록 빠름).
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # 자원 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()