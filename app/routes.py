from app import app
import os
import urllib.request
from flask import flash, request, redirect, render_template , Response ,jsonify
from werkzeug.utils import secure_filename
import sys

ALLOWED_EXTENSIONS = set(['mp4','wmv','mov','jpg'])

@app.route('/')
def upload():
	return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/upload',methods =['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            status_code = Response(status=204)
            return status_code
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            seperate_frame()
            string = string_represent()
            thai = FKNN(string)
            data = {
                'result' : thai,
                'status_code' : '200'
            }
            return jsonify(data)
        else:
            status_code = Response(status=406)
            return status_code
@app.route('/test',methods =['POST'])
def upload_file2():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            status_code = Response(status=204)
            return status_code
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            data = {
                'result' : "thai",
                'status_code' : '200'
            }
            return jsonify(data)
        else:
            status_code = Response(status=406)
            return status_code

def seperate_frame():
    import numpy as np
    import os
    import cv2
    import time
    source = "/root/senior-project/app/upload_video/"
    video = os.listdir(source)
    print("Seperate frame from Video")
    cap = cv2.VideoCapture(source+video[0])
    total_frames = cap.get(7)
    # print(total_frames)
    gap_time = total_frames/14
    # print(gap_time)
    success,image = cap.read()
    count = 0
    target_folder = "/root/senior-project/app/video_frame/"
    os.chdir(target_folder) 
    print("currently seperate frame")
    while success:
        if(count > 0):
            cap.set(cv2.CAP_PROP_POS_MSEC,(count*gap_time*35))
            # dim=(720,576)
            # resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite("%d.jpg" % count, image)
            success,image = cap.read()
            print('Read a new frame: ', success)
        count+=1
    
    os.chdir('/root/senior-project/app/')
    os.remove(source+video[0])
    print("Video Removed!")
def string_represent():
    import cv2
    import numpy as np
    import os
    import time
    import re
    print("String Representation Time")
    source  = "/root/senior-project/app/video_frame/"
    gesture_source = '/root/senior-project/app/gesture_lib_new/'
    datalist = os.listdir(source)
    gesture_lib = os.listdir(gesture_source)
    datalist.sort(key=lambda f: int(re.sub('\D', '', f)))
    sift_threshold = 0.7
    sift = cv2.xfeatures2d.SIFT_create()
    lib_kpdes = []
    # print(datalist)
    #-----------------------------------------------------------
    ### incase gesture is ready
    # with open("gesture_keypoint.p" , "rb") as f :
    #     lib_kpdes2 = cPickle.load(f)
    # lib_kpdes_new = []
    # for list_kpdes in lib_kpdes2:
    #     list_kp = []
    #     for kpdes in list_kpdes:
    #         kp_tuple = kpdes[0]
    #         des = kpdes[1]
    #         kp= []
    #         for point in kp_tuple:
    #             temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], 
    #                                             _response=point[3], _octave=point[4], _class_id=point[5]) 
    #             kp.append(temp)
    #         list_kp.append([kp,des])
    #     lib_kpdes_new.append(list_kp)
    # lib_kpdes = lib_kpdes_new
    #----------------------------------------------------------
    print("start SIFT gesture library")
    for index,folder in enumerate(gesture_lib):
    
        gesture_folder = os.listdir(gesture_source+folder)
        lib_kpdes_inloop = []
        for img in gesture_folder :
            gesture_img = cv2.imread(gesture_source+folder+'/'+img,1) # library
            kp2, des2 = sift.detectAndCompute(gesture_img, None)
            lib_kpdes_inloop.append([kp2,des2])
        lib_kpdes.append(lib_kpdes_inloop)
    print("start SIFT data")
    string = ''
    for source_folder in datalist:
        train_img = cv2.imread(source+source_folder,1) # train img
        kp1, des1 = sift.detectAndCompute(train_img, None)
        n_match_kp = np.zeros(len(gesture_lib))
        n_gesture_img = np.zeros(len(gesture_lib))
        for index,g_kpdes in enumerate(lib_kpdes): 
            kp_match = 0
            for kpdes in g_kpdes :
                kp2 = kpdes[0]
                des2 = kpdes[1]
                # FLANN parameters
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks=100)   # or pass empty dictionary

                flann = cv2.FlannBasedMatcher(index_params,search_params)
                matches = flann.knnMatch(des1,des2,k=2)

                # Need to draw only good matches, so create a mask
                matchesMask = [[0,0] for i in range(len(matches))]
                # ratio test as per Lowe's paper
                good_match = 0
                for i,(m,n) in enumerate(matches):
                    if m.distance < sift_threshold *n.distance:
                        matchesMask[i]=[1,0]
                        good_match += 1

                draw_params = dict(matchColor = (0,255,0),
                                    singlePointColor = (255,0,0),
                                    matchesMask = matchesMask,
                                    flags = 0)
                kp_match = kp_match + good_match
            n_gesture_img[index] = len(g_kpdes)
            n_match_kp[index] = kp_match
        argmax_match = np.argmax(n_match_kp/n_gesture_img)
        string_match = gesture_lib[argmax_match]
        string = string + string_match
        # print((n_match_kp / n_gesture_img)[30])
    print(string)
    #delete frame
    for i in range(len(datalist)):
        os.remove(source+datalist[i])
    print("frame Removed !!")
    return string

def levenshtein(seq1, seq2):
    import numpy as np
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
#     print (matrix)
    return (matrix[size_x - 1, size_y - 1])
def FKNN(string):
    import cv2
    import numpy as np
    import os
    import time
    import pickle
    print("start FKNN")
    with open("save_sc.p" , "rb") as f :
        sc = pickle.load(f)

    k_nearest = 12 #must less or equal than sub_class
    n_class = 10
    sub_class = 12
    m_constant=2
    sub_class = 12
    result = 0
    distance = []
    for i in range(n_class):
        for j in range(sub_class):
            distance.append((levenshtein(string,sc[i][j]) , i+1))
    sort_distance = sorted(distance,key=lambda tup:tup)
    result_class = np.zeros(n_class)
    for k in range(n_class):
        FKNN_sum = 0
        FKNN_divider_sum = 0 
        for m in range(k_nearest):
            if(k+1 == sort_distance[m][1]):
                uij = 1
            elif(k+1 != sort_distance[m][1]):
                uij = 0
            top = uij * pow((1/sort_distance[m][0]),1/(m_constant-1))
            FKNN_sum = FKNN_sum +top  
            FKNN_divider_sum = FKNN_divider_sum + pow((1/sort_distance[m][0]),1/(m_constant-1))
        result_class[k] =  FKNN_sum/FKNN_divider_sum
    result= np.argmax(result_class) 
    thai_language = ['พี่','ปู่','ย่า','ขอบคุณ','เข้าใจ','ผู้ชาย','ผู้หญิง','กตัญญู','คิดถึง','ยินดี']
    print(thai_language[result] , result)
    return thai_language[result]