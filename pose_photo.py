import cv2
import math
import time

def output_keypoints_photo( frame,net,proto_file, weights_file, threshold, BODY_PARTS):
    global points

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False,
                                       crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            # print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            # print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    return frame

def output_keypoints_with_lines_photo(POSE_PAIRS, frame):

    # 프레임 복사
#     frame_line = frame.copy()
    results={'shoulder1':0,"shou1der2":0,"waist1":0,"waist2":0,"neck":0}

    # 허리 
    if (points[1] is not None) and (points[8] is not None) and (points[9] is not None) and (points[12] is not None):
    	results['waist1']=calculate_degree_waist(point_1=points[1], point_2=points[8], point_3=points[9], point_4=points[12], frame=frame)
    
    # 허리 2
    if (points[12] is not None) and (points[9] is not None):
    	results['waist2']=calculate_degree_waist2(point_1=points[12], point_2=points[9], frame=frame)  

    # 목
    if (points[0] is not None) and (points[1] is not None) and (points[2] is not None) and (points[5] is not None):
    	results['neck']=calculate_degree_neck(point_1=points[0], point_2=points[1],point_3=points[2],point_4=points[5], frame=frame)
        
    # 어깨
    if (points[0] is not None) and (points[2] is not None) and (points[5] is not None):
    	results['shoulder1']=calculate_degree_shoulder(point_1=points[0], point_2=points[2],point_3=points[5], frame=frame)
    
    # 어깨 2
    if (points[5] is not None) and (points[2] is not None):
    	results['shoulder2']=calculate_degree_shoulder2(point_1=points[5], point_2=points[2], frame=frame)
    
    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            # Neck 과 MidHip 이라면 분홍색 선
            if part_a == 1 and part_b == 8:
                cv2.line(frame, points[part_a], points[part_b], (255, 0, 255), 3)
            else:  # 노란색 선
                cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
    
    # 포인팅 되어있는 프레임과 라인까지 연결된 프레임을 가로로 연결
#     frame_horizontal = cv2.hconcat([frame, frame_line])
    return frame,results



def calculate_degree_waist(point_1, point_2, point_3, point_4, frame):
    vector_a = [point_1[0] - point_2[0], point_1[1] - point_2[1]]
    vector_b = [point_3[0] - point_4[0], point_3[1] - point_4[1]]

    try:
        cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / ( ((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5) )
    except ZeroDivisionError:
        cos_theta = 1
        print("ZeroDivisionError")

#    cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / ( ((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5) )+1e-2

    print(cos_theta)
    if abs(cos_theta) > (2 ** 0.5) / 2:
        string = "Waist1-Forward Waist"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        print(f"[degree] ({string})")
        return 2
    else:
        string = "Waist1-Good"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255))
        print(f"[degree]({string})")
        return 1


def calculate_degree_waist2(point_1, point_2, frame): # p1 -> 12번, p2 -> 9번
    vector_a = [1, 0]
    vector_b = [point_2[0] - point_1[0], point_2[1] - point_1[1]]

    # cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / ( ((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5) ) +1e-2

    try:
        cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / ( ((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5) ) +1e-2
    except ZeroDivisionError:
        cos_theta = 1
        print("ZeroDivisionError")

    if abs(cos_theta) <= (3 ** 0.5) / 2:
        string ="Waist2-Asymmetry"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (25, 100, 25))
        print(f"[degree] ({string})")
        return 2
    else:
        string = "Waist2-Good"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (25, 100, 25))
        print(f"[degree] ({string})")
        return 1

def calculate_degree_neck(point_1, point_2, point_3, point_4, frame):
    vector_a = [point_1[0] - point_2[0], point_1[1] - point_2[1]]
    vector_b = [point_3[0] - point_4[0], point_3[1] - point_4[1]]

    # cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5)) +1e-2

    try:
        cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5)) +1e-2
    except ZeroDivisionError:
        cos_theta = 1
        print("ZeroDivisionError")

    # print(cos_theta)
    if abs(cos_theta) > (2 ** 0.5) / 2:
        string = "Neck-Forward Head"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0))
        print(f"[degree] ({string})")
        return 2
    else:
        string = "Neck-Good"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0))
        print(f"[degree] ({string})")
        return 1


def calculate_degree_shoulder(point_1, point_2,point_3, frame): #0,2,5
    vector_a = [point_3[0] - point_1[0], point_3[1] - point_1[1]]
    vector_b = [point_2[0] - point_1[0], point_2[1] - point_1[1]]

    # cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5))+1e-2

    try:
        cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5))+1e-2
    except ZeroDivisionError:
        cos_theta = 1
        print("ZeroDivisionError")


    print(cos_theta)
    # degree 가 45'보다 작으면 어깨가 숙여졌다고 판단
    if cos_theta <= -1/2:
        string = "Shoulder1-Round Shoulder"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))
        print(f"[degree]  ({string})")
        return 2
    else:
        string = "Shoulder1-Good"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))
        print(f"[degree] ({string})")
        return 1


def calculate_degree_shoulder2(point_1, point_2, frame): # p1 -> 5번, p2 -> 2번
    vector_a = [1, 0]
    vector_b = [point_2[0] - point_1[0], point_2[1] - point_1[1]]

    # cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / ( ((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5) )+1e-3

    try:
        cos_theta = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / ( ((vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5) * ((vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5) )+1e-3
    except ZeroDivisionError:
        cos_theta = 1
        print("ZeroDivisionError")

    if abs(cos_theta) <= (3 ** 0.5) / 2:
        string = "Shoulder2-Asymmetry"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 100, 255))
        print(f"[degree] ({string})")
        return 2
    else:
        string = "Shoulder2-Good"
        cv2.putText(frame, string, (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,100, 255))
        print(f"[degree]  ({string})")
        return 1

