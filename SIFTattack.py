import cv2
import numpy as np
import sys


# averaging perturbation.
def perturbation1(frame):
    sift=cv2.xfeatures2d.SIFT_create()
    keyp,desc=sift.detectAndCompute(frame,None)
    # img1=cv2.drawKeypoints(frame,keyp,None,color=0)

    # cv2.waitKey(0)
    #chosen kernel is 5x5
    k=2

    for i in keyp:
        point=getattr(i,"pt")
        point=(int(point[1]),int(point[0]))
        if (point[0]-k-1>0 and point[0]+k<frame.shape[0] ) and (point[1]-k-1>0 and point[1]+k<frame.shape[1]):
        # print(frame.shape)
            img_sub=frame[point[0]-k-1:point[0]+k,point[1]-k-1:point[1]+k]
            # print(point)
            img_sub=cv2.blur(img_sub,(7,7))
            frame[point[0]-k-1:point[0]+k,point[1]-k-1:point[1]+k]=img_sub
    return frame


#block2block perturbation
def perturbation2(frame):
    # frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    keyp,desc=sift.detectAndCompute(frame,None)

    k=2
    for i in keyp:
        point=getattr(i,"pt")
        point=(int(point[1]),int(point[0]))
        if (point[0]-1>0 and point[0]+1<frame.shape[0] ) and (point[1]-1>0 and point[1]+1<frame.shape[1]):
        # print(frame.shape)
            threshold = 50
            img_sub=frame[point[0]-1:point[0]+2,point[1]-1:point[1]+2]
            # print(point)
            a = np.array([np.sum(img_sub[0:2,0:2,:]),np.sum(img_sub[1:3,0:2,:]),np.sum(img_sub[0:2,1:3,:]), np.sum(img_sub[1:3,1:3,:])])
            i = np.argmax(a)
            if i!=0 and np.sum(img_sub[0:2,0:2,:])>=threshold:
                a = np.mean(img_sub[0:2,0:2,:])
                img_sub[0:2,0:2,:] = a*np.ones((2,2,3))
            if i!=1 and np.sum(img_sub[1:3,0:2,:])>=threshold:
                a = np.mean(img_sub[1:3,0:2,:])
                img_sub[1:3,0:2,:] = a*np.ones((2,2,3))
            if i!=2 and np.sum(img_sub[0:2,1:3,:])>=threshold:
                a = np.mean(img_sub[0:2,1:3,:])
                img_sub[0:2,1:3,:] = a*np.ones((2,2,3))
            if i!=3 and np.sum(img_sub[1:3,1:3,:])>=threshold:
                a = np.mean(img_sub[1:3,1:3,:])
                img_sub[0:2,0:2,:] = a*np.ones((2,2,3))

            frame[point[0]-1:point[0]+2,point[1]-1:point[1]+2,:]=img_sub
    return frame


#Pixel2Pixel perturbation
def perturbation3(frame):
    # frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    keyp,desc=sift.detectAndCompute(frame,None)
    # img1=cv2.drawKeypoints(frame,keyp,None,color=0)

    # cv2.waitKey(0)
    k=2
    for i in keyp:
        point=getattr(i,"pt")
        point=(int(point[1]),int(point[0]))
        if (point[0]-k-1>0 and point[0]+k<frame.shape[0] ) and (point[1]-k-1>0 and point[1]+k<frame.shape[1]):
        # print(frame.shape)
            img_sub=frame[point[0]-k-1:point[0]+k,point[1]-k-1:point[1]+k]
            # print(point)
            height, weight,_ = img_sub.shape
            for i in range(height):
                for j in range(weight):
                    if point[0]-k-1+(i-1)>=0 and point[0]-k-1+(i)<frame.shape[0] and point[1]-k-1+(j-1)>=0 and point[1]-k-1+(j)<frame.shape[1]:
                    # try:
                        img_sub[i,j] = frame[point[0]-k-1+(i-1),point[1]-k-1+(j)]+frame[point[0]-k-1+(i),point[1]-k-1+(j-1)]
                        img_sub[i,j] = img_sub[i,j]/2
                    # except :
                    #     print("hi")
            frame[point[0]-k-1:point[0]+k,point[1]-k-1:point[1]+k]=img_sub
    # return cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    return frame

#perturbs every frame of the selected video and saves a new video in .avi format.
def main(video_name,video_loc='Videos/'):
    videoCap=cv2.VideoCapture(video_loc+video_name)
    success,img=videoCap.read()
    # print(success)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fps=int(videoCap.get(cv2.CAP_PROP_FPS))
    print(fps)
    height = len(img)
    width = len(img[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filen=video_name.split(".")[0]
    outputVideo = cv2.VideoWriter('summary/'+filen+'attack.avi',fourcc,fps,(width,height))
    # inputVideo = cv2.VideoCapture(fileName)
    # outputVideo.write(perturbation(img))
    i=1
    s=1
    while success:
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        chh=perturbation3(img)
        outputVideo.write(chh)
        # print(chh.shape)
        success,img=videoCap.read()
        i+=1
        print(i)
        s+=1
    cv2.destroyAllWindows()
    outputVideo.release()



# import copy
# def main1(video="Videos/10.mp4"):
#     videoCap=cv2.VideoCapture(video)
#     success,frame=videoCap.read()
#     frame=cv2.resize(frame,None,fx=1/2,fy=1/2)
#     cv2.imwrite("0.png",frame)
#     cv2.imwrite("1.png",perturbation1(copy.copy(frame)))
#     cv2.imwrite("2.png",perturbation2(copy.copy(frame)))
#     cv2.imwrite("3.png", perturbation3(copy.copy(frame)))
vids=[0,1,4,6,7,9,10,11]
vidd=[8]
if __name__=='__main__':
    for i in vidd:
        main(str(i)+'.avi')
    # main1()
