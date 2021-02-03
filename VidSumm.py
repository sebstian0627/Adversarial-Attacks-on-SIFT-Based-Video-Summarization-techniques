import cv2
import sys
import os
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import namedtuple
import time


# fileName = "test.mp4"
folderName ="attack_summary_B2B/"


motionThreshold = 10

ColorMoments = namedtuple('ColorMoments', ['mean', 'stdDeviation', 'skewness'])


Shot = namedtuple('Shot', ['shotNumber', 'startingFrame', 'endingFrame', 'keyFrames', 'avgEntropyDiff', 'avgMotion'])

shotSifts = []

def rgb2gray(rgb):
    # return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

def getEntropy(histogram, totalPixels):
    entropy = 0
    for pixels in histogram:
        if pixels != 0:
            prob = float (pixels / totalPixels)
            entropy -= prob * math.log(prob, 2)
            # print prob
    return entropy

def getColorMoments(histogram, totalPixels):
    sum = 0
    for pixels in histogram:
        sum += pixels
    mean = float (sum / totalPixels)
    sumOfSquares = 0
    sumOfCubes = 0
    for pixels in histogram:
        sumOfSquares += math.pow(pixels-mean, 2)
        sumOfCubes += math.pow(pixels-mean, 3)
    variance = float (sumOfSquares / totalPixels)
    stdDeviation = math.sqrt(variance)
    avgSumOfCubes = float (sumOfCubes / totalPixels)
    skewness = float (avgSumOfCubes**(1./3.))
    return ColorMoments(mean, stdDeviation, skewness)

def getHistogramDiff(currHistogram, prevHistogram):
    diff = 0
    for i in range(len(currHistogram)):
        diff += math.pow(currHistogram[i]-prevHistogram[i], 2)
    if diff==0:
        return 1
    else:
        return diff

def getHistogramRatio(currHistogramDiff, prevHistogramDiff):
    ratio = float (currHistogramDiff / prevHistogramDiff)
    if ratio<1:
        ratio = 1/ratio
    return ratio

def getEuclideanDistance(currColorMoments, prevColorMoments):
    distance = math.pow(currColorMoments.mean - prevColorMoments.mean, 2) + math.pow(currColorMoments.stdDeviation - prevColorMoments.stdDeviation, 2) + math.pow(currColorMoments.skewness - prevColorMoments.skewness, 2)
    return distance

def getMotion(currImage, prevImage):
    motion = 0
    for i in range(len(currImage)):
        for j in range(len(currImage[i])):
            if int (currImage[i][j]) - int (prevImage[i][j]) > motionThreshold:
                motion += 1
    motion = float (motion / (i+1)*(j+1))
    return motion

def getSift(img):
    detector = cv2.xfeatures2d.SIFT_create(200)
    kp, des = detector.detectAndCompute(img, None)
    return des

def getMotionSift(img1, img2, ind):
    if ind==1:
        des1 = getSift(img1)
        shotSifts.append(des1)
    else:
        des1 = shotSifts[ind-1]
    des2 = getSift(img2)
    shotSifts.append(des2)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # des1.convertTo(des1,CV_32F)
    # des2.convertTo(des2,CV_32F)
    # if len(list(des1))==0 or len(list(des2))==0:
    #     return 0
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return 0
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    sumd = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            sumd += m.distance
            matchesMask[i]=[1,0]
    return sumd

def sortShots(shots):
    maxKeyFrames = -1
    maxAvgEntropyDiff = -1
    maxAvgMotion = -1
    for shot in shots:
        if shot.keyFrames > maxKeyFrames :
            maxKeyFrames = shot.keyFrames
        if shot.avgEntropyDiff > maxAvgEntropyDiff :
            maxAvgEntropyDif = shot.avgEntropyDiff
        if shot.avgMotion > maxAvgMotion :
            maxAvgMotion = shot.avgMotion

    weights = []
    for shot in shots:
        weight = shot.keyFrames / float(maxKeyFrames) + shot.avgEntropyDiff / float(maxAvgEntropyDiff) + shot.avgMotion / float(maxAvgMotion)
        weights.append((shot.shotNumber, weight))

    print ('Unsorted Weights -', weights)
    weights.sort(reverse=True, key=lambda x: x[1])
    print ('Sorted Weights -', weights)
    order = [int(weight[0]) for weight in weights]
    print( 'Order -', order)
    return order

def save_keyframes(frame_indices, summary_frames):
    global skipFrames
    print ("Saving frame indices")

    out_file=open(folderName+"/frame_indices_"+str(skipFrames)+".txt",'w')
    for idx in frame_indices:
        out_file.write(str(idx*skipFrames)+'\n')
    print ("Saved indices")

    print ("Saving frames")
    cmd="rm -r "+folderName+"/_keyframes_/"
    os.system(cmd)
    cmd="mkdir "+folderName+"_keyframes_"
    for i,frame in enumerate(summary_frames):
        cv2.imwrite(str(folderName)+"/_keyframes_/frame%d.jpg"%i, frame)
    print ("Frames saved")


def main(fileName,folderName ="summary/",skipFrames=1,motionThreshold=10):
    videoCap = cv2.VideoCapture(fileName)
    #fps = videoCap.get(cv2.CAP_PROP_FPS)
    # fps=60
    fps=int(1.4*videoCap.get(cv2.CAP_PROP_FPS))
    print ("Frames per second: ", fps,fileName)

    entropy = []
    histogramDiff = []
    histogramRatio = []
    entropyDiff = []
    euclideanDistance = []
    motion = []

    t0 = time.clock()

    i = 0
    #for a in range(skipFrames):
    #    success, image = videoCap.read()
    success, image = videoCap.read()
    print(success)
    height = len(image)
    #height = 1080
    width = len(image[0])
    #width = 1920
    # image = image[int(0.25*height):int(0.75*height)]
    totalPixels = width * height
    while success:
        # print width, height
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # plt.imshow(grayImage, cmap = plt.get_cmap('gray'))
        # plt.show()
        # print grayImage.shape
        # print grayImage
        histogram = cv2.calcHist([grayImage],[0],None,[256],[0,256])
        # print len(histogram)
        entropy.append( getEntropy(histogram, totalPixels) )
        # print entropy[i]
        colorMoments = getColorMoments(histogram, totalPixels)
        # print colorMoments
        if i==0:
            histogramDiff.append(4000000)
            histogramRatio.append(200)
            entropyDiff.append(0)
            euclideanDistance.append(0)
            motion.append(0)
        else:
            histogramDiff.append( getHistogramDiff(histogram, prevHistogram) )
            if i==1:
                histogramRatio.append(1)
            else:
                histogramRatio.append( getHistogramRatio(histogramDiff[i], histogramDiff[i-1]) )
            entropyDiff.append( abs(entropy[i] - entropy[i-1]))
            euclideanDistance.append( getEuclideanDistance(colorMoments, prevColorMoments) )
            motion.append( getMotionSift(grayImage, prevGrayImage, i) )

        prevHistogram = histogram
        prevGrayImage = grayImage
        prevColorMoments = colorMoments

        i += 1
        #for a in range(skipFrames):
        #    success, image = videoCap.read()
        success, image = videoCap.read()
        # image = image[int(0.25*height):int(0.75*height)]
        # print( i,fileName)

        # Uncomment this for breaking early i.e. 100 frames
        # if i==100:
        #     break

    meanEntropyDiff = sum(entropyDiff[1:]) / float(len(entropyDiff)-1)
    meanHistogramRatio = sum(histogramRatio[1:]) / float(len(histogramRatio)-1)
    meanEuclideanDistance = sum(euclideanDistance[1:]) / float(len(euclideanDistance)-1)

    thresholdEntropyDiff = meanEntropyDiff
    thresholdHistogramRatio = meanHistogramRatio
    thresholdEuclideanDistance = meanEuclideanDistance

    totalFrames = i

    motionSum = 0
    entropyDiffSum = 0

    shots = []
    shotNumber = 0
    prevFrame = 0
    keyFrames = 0
    keyFramesArray = [0] * totalFrames
    for i in range(totalFrames):
        if euclideanDistance[i] > thresholdEuclideanDistance:
            keyFrames += 1
            keyFramesArray[i] = 1

        entropyDiffSum += entropyDiff[i]
        motionSum += motion[i]

        if entropyDiff[i] > thresholdEntropyDiff and histogramRatio[i] > thresholdHistogramRatio:
            if i<= prevFrame+25 and shotNumber!=0:
                currShot = shots[shotNumber-1]
                numberOfFrames = currShot.endingFrame - currShot.startingFrame + 1
                newAvgEntropyDiff = ((currShot.avgEntropyDiff * numberOfFrames) + entropyDiffSum) / (i - currShot.startingFrame)
                newAvgMotion = ((currShot.avgMotion * numberOfFrames) + motionSum) / (i - currShot.startingFrame)
                shots[shotNumber-1] = Shot(currShot.shotNumber, currShot.startingFrame, i-1, currShot.keyFrames + keyFrames, newAvgEntropyDiff, newAvgMotion)
            else:
                avgEntropyDiff = entropyDiffSum / float(i - prevFrame)
                avgMotion = motionSum / float(i - prevFrame)
                shots.append(Shot(shotNumber, prevFrame, i-1, keyFrames, avgEntropyDiff, avgMotion))
                shotNumber += 1

            keyFrames = 0
            motionSum = 0
            entropyDiffSum = 0
            prevFrame = i

    # Adding the last shot
    if i!=prevFrame:
        avgEntropyDiff = entropyDiffSum / float(i - prevFrame)
        avgMotion = motionSum / float(i - prevFrame)
        shots.append(Shot(shotNumber, prevFrame, i-1, keyFrames, avgEntropyDiff, avgMotion))
        shotNumber += 1

    shotsOrder = sortShots(shots)
    trailerFrames = totalFrames*4//5
    writeBit = [0] * len(shots)
    for shotNo in shotsOrder:
        shot = shots[shotNo]
        shotFrames = shot.endingFrame - shot.startingFrame + 1
        if shotFrames < trailerFrames:
            writeBit[shotNo] = 1
            trailerFrames -= shotFrames
        if trailerFrames<=0:
            break

    print (width, height)
    fileName1=fileName.split("/")[-1]
    filen=fileName1.split(".")[0]
    print(filen, fileName)
    fps=videoCap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    outputVideo = cv2.VideoWriter(folderName+'/summary'+filen+'.avi',fourcc,fps,(width,height))
    inputVideo = cv2.VideoCapture(fileName)
    required_frames=[]
    for i in range(len(shots)):
        if writeBit[i]==1:
            j = shots[i].startingFrame * skipFrames
            while j<= shots[i].endingFrame * skipFrames:
                # required_frames.append(j)
                success, image = inputVideo.read()
                if success:
                    # print ('writing')
                    # plt.imshow(image)
                    # plt.show()
                    # print("hi")
                    required_frames.append(j)
                    outputVideo.write(image)
                else:
                    print( "Error in reading video")
                j += 1
        else:
            j = shots[i].startingFrame * skipFrames
            while j<= shots[i].endingFrame * skipFrames:
                success, image = inputVideo.read()
                if not success:
                    print( "Error in reading video")
                j += 1

    cv2.destroyAllWindows()
    return required_frames
    outputVideo.release()

    print ("Generating summary frames")
    summary_frames=[]
    keyFramesIndices=[]

    for i in range(len(keyFramesArray)):
        if(keyFramesArray[i]==1):
            cmd="cp "+folderName+"/allFrames/image"+str(i)+".jpg "+folderName+"/keyFrames/"
            os.system(cmd)
            keyFramesIndices.append(i)
            summary_frames.append(cv2.imread(folderName+"/allFrames/image"+str(i)+".jpg",-1))
    print ("Generated Summary !")
    save_keyframes(keyFramesIndices,summary_frames)



    print ('Write Bit -', writeBit)

    print ('Time taken to run =', time.clock() - t0, 'seconds')

    print ('len(Shots) -' , len(shots), '\n')
    print ('Shots -' , shots, '\n')
    print ('len(Entropy) -', len(entropy), '\n')
    #print 'Entropy -', entropy, '\n'
    print ('len(HistogramDiff) -', len(histogramDiff), '\n')
    #print ('HistogramDiff -', histogramDiff, '\n')
    print ('len(HistogramRatio) -', len(histogramRatio), '\n')
    #print 'HistogramRatio -', histogramRatio, '\n'
    print ('len(EntropyDiff) -', len(entropyDiff), '\n')
    #print 'EntropyDiff -', entropyDiff, '\n'
    print ('len(EuclideanDistance) -', len(euclideanDistance), '\n')
    #print 'EuclideanDistance -', euclideanDistance, '\n'
    print ('len(Motion) -', len(motion), '\n')
    #print 'Motion -', motion, '\n'
    #print 'len(MeanEntropyDiff) -', len(meanEntropyDiff), '\n'
    print ('MeanEntropyDiff -', meanEntropyDiff, '\n')
    #print 'len(MeanHistogramRatio) -', len(meanHistogramRatio), '\n'
    print ('MeanHistogramRatio -', meanHistogramRatio, '\n')
    #print 'len(MeanEuclideanDistance) -', len(meanEuclideanDistance), '\n'
    print ('MeanEuclideanDistance -', meanEuclideanDistance, '\n')
    print ('len(keyFramesArray) - ', len(keyFramesArray), '\n')
    print ('# of keyFrames - ', sum(keyFramesArray), '\n')
    print ('keyFramesArray - ', keyFramesArray, '\n')

# import os
def ground_truth_frames_generator(l,fps):
    if len(l)==0:
        return [0]
    else:
        result=[]
        for j in range(len(l)):
            (x,y)=l[j]
            for k in range(x*fps,y*fps+1):
                result.append(k)
        return result

def result_finder(l1,l2,gt,totalframes):
    s1=0
    s2=0
    for i in gt:
        s1+=len(list(set(l1)&set(i)))/len(i)
        s2+=len(list(set(l2)&set(i)))/len(i)
    return (s1/len(gt),s2/len(gt))

#define it for yourself
if __name__ == '__main__':
    # for i in range(6,13):
        vids=[10]
        # vids=[1]
        frames_0=[]
        resultss=[]
        for i in vids:
            # try:


            cap = cv2.VideoCapture("Videos/" + str(i) + ".mp4")
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            gt_frames_1 = ground_truth_frames_generator([(0,12),(15,30)],
                                                        fps)
            gt_frames_2 = ground_truth_frames_generator([(6,32)] ,fps)
            gt_frames_3 = ground_truth_frames_generator([(8,32)], fps)
            gt_frames_4 = ground_truth_frames_generator([(12,32)], fps)
            # frames_0=main("Videos/"+str(i)+".mp4")
            #
            # frames_1=main("Attack1/"+str(i)+"attack.avi")
            # (sr1, ar1) = result_finder(frames_0, frames_1, [gt_frames_1, gt_frames_2,gt_frames_3,gt_frames_4],
            #                            tot_frame)
            frames_2=main("Attack2/"+str(i)+"attack.avi")
            frames_3=main("Attack3/"+str(i)+"attack.avi")

            (ar2,ar3)=result_finder(frames_2,frames_3,[gt_frames_1,gt_frames_2,gt_frames_3,gt_frames_4],tot_frame)

            print(ar2,ar3)
            # except:
            #     os.mkdir("i")
        # vids=[10,11]
        # for i in vids:
        # print(results)
        # main("Attack2/"+str(8)+"attack.avi")
        # main("Attack1/"+str(10)+"attack.avi")
                # os.mkdir("attack"+str(i))
