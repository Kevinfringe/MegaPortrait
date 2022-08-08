import cv2 as cv
import numpy as np
import os


acted_anger_source_path = "./videos_zakir_80/videos_zakir_80/AA1.mp4"
genuine_anger_driver_path = "./videos_zakir_80/videos_zakir_80/GA1.mp4"
source_imgs_path = "./source_img"
driver_imgs_path = "./driver_img"

print("current work dir" + os.getcwd())


def capture_frames(path, flag, n=512):
    '''

    :param n:
    :param path:
    :param flag: if flag = 1, then it generate source img, otherwise, generate driver img.
    :return:
    '''
    counter = 1

    cap = cv.VideoCapture(path)
    #cap.set(cv.CAP_PROP_FPS, 60.0)
    print("Current sampling FPS is : " + str(cap.get(cv.CAP_PROP_FPS)))



    # ref: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    while cap.isOpened():
        print("Entered while loop.")
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if flag == 1:
            filename = "A"
        else:
            filename = "G"


        img = cv.resize(frame, (n, n))
        cwd = os.getcwd()
        if flag == 1:
            os.chdir(source_imgs_path)
        else:
            os.chdir(driver_imgs_path)
        filename = filename + str(counter) + ".jpg"
        print("Image shape is: " + str(img.shape))
        cv.imwrite(filename=filename, img=img)
        os.chdir(cwd)

        counter += 1

    cap.release()


capture_frames(acted_anger_source_path, flag=1)
capture_frames(genuine_anger_driver_path, flag=0)