

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
    print(thai_language[result])
