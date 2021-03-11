import cv2
import numpy as np
import os
import time
import re
source  = "/root/senior-project/app/video_frame/"
gesture_source = '/root/senior-project/app/gesture_lib/'
datalist = os.listdir(source)
gesture_lib = os.listdir(gesture_source)
datalist.sort(key=lambda f: int(re.sub('\D', '', f)))
sift_threshold = 0.7
sift = cv2.xfeatures2d.SIFT_create()
lib_kpdes = []
print(datalist)
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
for index,folder in enumerate(gesture_lib):
    gesture_folder = os.listdir(gesture_source+folder)
    lib_kpdes_inloop = []
    for img in gesture_folder :
        gesture_img = cv2.imread(gesture_source+folder+'/'+img,1) # library
        kp2, des2 = sift.detectAndCompute(gesture_img, None)
        lib_kpdes_inloop.append([kp2,des2])
    lib_kpdes.append(lib_kpdes_inloop)
# print(len(lib_kpdes))
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