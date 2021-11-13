from flask import Flask,render_template,Response,redirect,url_for,request, flash,stream_with_context
import cv2
import os
import time
import re
import ast
from flask.json import jsonify
from flask_restful import Resource, Api
import pandas as pd
import threading 
import multiprocessing
from pose_live import output_keypoints,output_keypoints_with_lines
from pose_photo import output_keypoints_photo,output_keypoints_with_lines_photo
from werkzeug.utils import secure_filename
import argparse
import json
app=Flask(__name__)
app.secret_key="disuo"
api=Api(app)
UPLOAD_FOLDER=os.getcwd()+'\static\images'
ALLOWED_EXTENSIONS=set(['png','jpg','gif'])
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

headings=("과","병원명","도","시","동")

data=None
address={}
# Initialize output variables and define a multithreaded lock to prevent simultaneous access from multiple browsers or pages
# When the video stream is output
outputFrame = None
results=None
lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search_hospital',methods=['POST','GET'])
def hospital():
    global data,headings
    global address
    if request.method=='POST':
        address=request.data.decode("UTF-8")
        address=ast.literal_eval(address)

        print(address)
        han=pd.read_csv('./static/han.csv',encoding='cp949')
        jung=pd.read_csv('./static/jung.csv',encoding='cp949')
        tong=pd.read_csv('./static/tong.csv',encoding='cp949')


        han_result=han[ (han['시'].str.contains(address[1], case=False)) & (han['동']==address[2])]
        # print(han_result)
        han_result=list(han_result.itertuples(index=False, name=None))


        tong_result=tong[(tong['시'].str.contains(address[1], case=False))&(tong['동']==address[2])]
        # print(tong_result)
        tong_result=list(tong_result.itertuples(index=False, name=None))


        jung_result=jung[ (jung['시'].str.contains(address[1], case=False))&(jung['동']==address[2])]
        # print(tong_result)
        jung_result=list(jung_result.itertuples(index=False, name=None))

        data=tuple(han_result+tong_result+jung_result) 

        html_text=''
        for d in data:
            tmp="<tr>\n"
            cnt=0
            for i in d:
                if cnt==1: 
                    temp="<td class='text-center' onclick='location.href="+'"https:/search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query='+i+'";'+"'>"+i+"</td>\n"
                else: 
                    temp="<td class='text-center'> "+i+"</td>\n"
                tmp=tmp+temp
                cnt+=1
            html_text=html_text+tmp+"</tr>\n"

        html_file = open('./static/html_file.html', 'w',encoding="UTF-8")
        html_file.write(html_text)
        html_file.close()


    print(data)
    return render_template('hospital.html',headings=headings,data=data)


@app.route('/static_ver')
def static_ver():    
    file_path='static/images/img-02.jpg'
    return render_template('static.html',file_path=file_path,result=None)


@app.route('/static_ver/<file_path>')
def static_ver2(file_path):   
    file_path='/static/images/'+file_path
    result=request.args.get('results_photo')
    print(type(result))
    return render_template('static.html',file_path=file_path,result=result)


@app.route('/uploader',methods=['POST','GET'])
def uploader_file():
    if request.method=='POST':
        file=request.files['file']
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            results_photo=gen_photo(filename,os.path.join(app.config['UPLOAD_FOLDER'],filename))
            print(type(results_photo))
            return redirect(url_for('static_ver2',file_path=filename[0:-3]+"_pose.jpg",results_photo=results_photo, code=307))


@app.route('/reload')
def reload():
    return render_template('html_file.html')

def detect_motion():

    global outputFrame,lock,results

    BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

    POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                        [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                        [11, 24], [22, 24], [23, 24]]

    # 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
    protoFile_body_25 = "C:\\openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt"

    # 훈련된 모델의 weight 를 저장하는 caffemodel 파일
    weightsFile_body_25 = "C:\\openpose-master\\models\\pose\\body_25\\pose_iter_584000.caffemodel"

    # 키포인트를 저장할 빈 리스트
    points = []

    net = cv2.dnn.readNetFromCaffe(protoFile_body_25, weightsFile_body_25)
    
    # GPU 사용
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv2.VideoCapture(1)
    capture.set(3,800)
    capture.set(4,800)
    
    if not capture.isOpened():
        capture=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise IOError("Cannot open webcame") 

    prev_time = 0
    FPS = 10
    
    while cv2.waitKey(1) <0:

        current_time = time.time() - prev_time
        ret, frame_boy = capture.read()
        if not ret:
            cv2.waitKey()
            break
        
        elif ret and current_time>1./FPS: 
        
            prev_time = time.time()
            
            frame_boy = output_keypoints(frame=frame_boy, net=net,proto_file=protoFile_body_25, weights_file=weightsFile_body_25, threshold=0.1, BODY_PARTS=BODY_PARTS_BODY_25)
            frame_boy,results_live = output_keypoints_with_lines(frame=frame_boy, POSE_PAIRS=POSE_PAIRS_BODY_25)
            # ret,buffer=cv2.imencode('.jpg',frame_boy)
            # frame=buffer.tobytes()

            with lock:
                # outputFrame=frame.copy()
                outputFrame=frame_boy.copy()
                results=results_live

def gen():

    global outputFrame, lock
    # Traverse the frames of the output video stream
    while True:
      # Wait until the thread lock is acquired
        with lock:
         # Check whether there is content in the output. If there is no content, skip this process
         if outputFrame is None:
            continue

         # Compress the output to jpeg format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        frame=encodedImage.tobytes()
         # Make sure the output is compressed correctly
        if not flag:
            continue

        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen2():
    global results,lock
    # Traverse the frames of the output video stream
    while True:
      # Wait until the thread lock is acquired
        with lock:
         # Check whether there is content in the output. If there is no content, skip this process
            if results is None:
                continue
        
        yield "바보"


def gen_photo(filename,file_path):

    BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

    POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                        [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                        [11, 24], [22, 24], [23, 24]]

    # 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
    protoFile_body_25 = "C:\\openpose-master\\models\\pose\\body_25\\pose_deploy.prototxt"

    # 훈련된 모델의 weight 를 저장하는 caffemodel 파일
    weightsFile_body_25 = "C:\\openpose-master\\models\\pose\\body_25\\pose_iter_584000.caffemodel"

    # 키포인트를 저장할 빈 리스트
    points = []

    net = cv2.dnn.readNetFromCaffe(protoFile_body_25, weightsFile_body_25)
    
    # GPU 사용
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    frame_boy = cv2.imread(file_path)
    frame_boy = output_keypoints_photo(frame=frame_boy, net=net,proto_file=protoFile_body_25, weights_file=weightsFile_body_25, threshold=0.1, BODY_PARTS=BODY_PARTS_BODY_25)
    frame_boy,results_photo = output_keypoints_with_lines_photo(frame=frame_boy, POSE_PAIRS=POSE_PAIRS_BODY_25)

    cv2.imwrite('./static/images/'+filename[0:-3]+'_pose.jpg',frame_boy)

    return results_photo


@app.route('/live_ver')
def live_ver():
    return redirect(url_for('timer', num=25*60))

@app.route('/get_result', methods = ['GET','POST'])
def get_result():
    # return jsonify(results)
    return Response(stream_with_context(gen2())) #, content_type='text/event_stream'
    # return gen2()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_ver')
def live():
    return redirect(url_for('timer', num=25*60))

@app.route('/live_ver/<int:num>s')
@app.route('/live_ver/<int:num>')
def timer(num):
    # Start target detection thread
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()
    return render_template('live.html', num=num)

@app.route('/live_ver/<int:num>m')
def minutes(num):
    return redirect(url_for('timer', num=num*60))

@app.route('/live_ver/<int:num>h')
def hours(num):
    return redirect(url_for('timer', num=num*3600))

@app.route('/live_ver/custom', methods=['GET', 'POST'])
def custom():
    time = request.form.get('time', 180)
    # use re to validate input data
    m = re.match('\d+[smh]?$', time)
    if m is None:
        flash(u'시간을 다음과 같은 형식으로 입력해주세요 34、20s、15m、2h')
        return redirect(url_for('index'))
    if time[-1] not in 'smh':
        return redirect(url_for('timer', num=int(time)))
    else:
        type = {'s': 'timer', 'm': 'minutes', 'h': 'hours'}
        return redirect(url_for(type[time[-1]], num=int(time[:-1])))

if __name__ == "__main__":



    app.run(host='0.0.0.0',port=5000) #http://121.168.117.223:5000/

