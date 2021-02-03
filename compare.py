import cv2
import numpy as np
import os
import glob
from skimage.metrics import structural_similarity

#calculates ssim score framewise and average it over all frames
def main(source,attack):
    cap1=cv2.VideoCapture(source)
    cap2=cv2.VideoCapture(attack)
    total_frames=0
    success1,frame1=cap1.read()
    success2,frame2=cap2.read()
    varr=0
    while success1 and success2:
        total_frames+=1
        varr+=compare(frame1,frame2)
        success1,frame1=cap1.read()
        success2,frame2=cap2.read()
        # print(total_frames)
    return varr/total_frames


#returns structural_similarity between 2 images (0-1)
def compare(img1, img2):
    # img1 = cv2.imread(ImageAPath)          # queryImage
    # img2 = cv2.imread(ImageBPath)
    # image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)          # trainImage
    image1=img1
    image2=img2

    score, diff = structural_similarity(image1, image2, full=True,  multichannel=True)
    return score

#averages over all videos.
if __name__=="__main__":
    vids=[1,4,6,7,9,10,11 ] #define it 
    sum=0
    for i in vids:
        source="Videos/"+str(i)+".mp4"
        attack="Attack3/"+str(i)+"attack.avi"
        print("hi")
        x=main(source,attack)
        print(i)
        sum+=x
    sum+=main("Videos/"+str(8)+".avi","Attack3/8attack.avi")
    print(sum/(len(vids)+1))
