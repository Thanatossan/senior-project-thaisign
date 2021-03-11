def seperate_frame():
    import numpy as np
    import os
    import cv2
    source = "/root/senior-project/app/upload_video/"
    video = os.listdir(source)
    print(source)
    cap = cv2.VideoCapture(source+video[0])
    total_frames = cap.get(7)
    print(total_frames)
    gap_time = total_frames/14
    print(gap_time)
    success,image = cap.read()
    count = 0
    target_folder = "/root/senior-project/app/video_frame/"
    os.chdir(target_folder)
     
    while success:
        if(count > 0):
            cap.set(cv2.CAP_PROP_POS_MSEC,(count*gap_time*35))
            cv2.imwrite("%d.jpg" % count, image)
            success,image = cap.read()
            # print('Read a new frame: ', success)
        count+=1
    os.chdir('/root/senior-project/app/')
    os.remove(source+video[0])
    print("Video Removed!")
