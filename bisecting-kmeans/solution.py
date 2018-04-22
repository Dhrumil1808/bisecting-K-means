import operator
from math import exp, log
from random import randint
from sklearn.decomposition import TruncatedSVD
import time
import random
from pathlib import Path
import pickle
import optimized

import matplotlib.pyplot as plt
pickleFile = Path("./pickleFile2.txt")
pickleFile1 = Path("./pickleFile1.txt")

# if my_file.is_file():
#print("Start time is:" + str(startTime))
with open('./train.dat', 'r') as fh:
    lines = fh.readlines()
train = []
min = 9999
max = 0
docFreq = {}
wordsInDoc = []
pointToClusterMap = {}

if pickleFile.is_file():
    with open("pickleFile2.txt", "rb") as myFile:
        print("Loaing pickle!!")
        tfidfOri = pickle.load(myFile)
else:
    print("Creating pickle!!")
    print("Creating dictionary!!")
    for l in lines:
        l = l.split()
        wordsInDoc.append((len(l)) / 2)
        i = 0
        dict = {}
        for i in range(len(l) - 1):
            if (i + 1) % 2 == 0 and i != 0:
                continue
            if l[i] not in dict:
                if int(l[i]) > max:
                    max = int(l[i])
                if int(l[i]) < min:
                    min = int(l[i])
                dict[l[i]] = l[i + 1]
            else:
                dict[l[i]] += l[i + 1]

        train.append(dict)
    # print(train)
    for i in range(min, max):
        for l in train:
            if str(i) in l:
                if i in docFreq:
                    docFreq[i] += 1
                else:
                    docFreq[i] = 1
    tfidf = []
    j = 0
    print("Calculating tfidf!!")
    for l in train:
        list = []
        for i in range(min, max):
            if str(i) in l:
                tf = (int(l[str(i)]) / wordsInDoc[j])
                idf = log(len(train) / docFreq[i])
                list.append(tf / idf)
            else:
                list.append(0)
        tfidf.append(list)
        j += 1

    svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42).fit(tfidf)
    tfidf = svd.transform(tfidf)
    with open("pickleFile2.txt", "wb") as myFile:
        pickle.dump(tfidf, myFile)
proximityMatrix = []
i = 0
j = 0
numOfClusters=7
ssePerClusterIteration=[]
sseVsCluster=[]
totalTime1=[]
allx1=[]
totalDatatSet=1580
for x in range(8):
    startTime = time.time()

    tfidf=tfidfOri[:totalDatatSet][:]
    allx1.append(totalDatatSet)

    '''if pickleFile1.is_file():
        with open("pickleFile1.txt", "rb") as myFile:
            print("Loaing pickle!!")
            proximityMatrix = pickle.load(myFile)
    else:
        print("Creating pickle!!")
        print("Calculating proximity matrix!!")
        for i in range(len(tfidf)):
            list = []
            for j in range(len(tfidf)):
                if i == j:
                    d = 0
                else:
                    d = calulateDist(tfidf[i], tfidf[j])
                list.append(d)
            proximityMatrix.append(list)
        with open("pickleFile1.txt", "wb") as myFile:
            pickle.dump(proximityMatrix, myFile)'''
    print("Starting clustering!!")
    clusterIndex=0
    for i in range(2):

        intialPoints = []

        for i in range(2):
            intialPoints.append(randint(0,totalDatatSet-1))

        clusters = {}
        for point in intialPoints:
            list = []
            list.append(point)
            clusters[point] = list
            pointToClusterMap[point] = point



    i = 0

    sse = {}


    def calculateSSE(list, mean):
        l=[]
        ans=0
        for j in range(len(tfidf[0])):
            l.append(0)
            for i in range(len(list)):
                l[j]+=(mean[j]-tfidf[list[i]][j])**2


        for a in l:
            ans+=a
        a/=len(tfidf[0])
        return a


    def calculateMean(list):
        l = []
        ans = 0
        for j in range(len(tfidf[0])):
            l.append(0)
            for i in range(len(list)):
                l[j] +=  tfidf[list[i]][j]
        return l


    def calculateDist(p1, p2):
        d = 0
        for i in range(len(p1)):
            d+=(p1[i]-p2[i])**2
        d=d**0.5
        return d


    def breakCluster(point):
        global clusterIndex,clusters,numOfClusters
        if len(clusters) == numOfClusters:
            return None
        list = clusters[point]
        breakPoints = []
        for i in range(2):
            r = random.choice(list)
            if r == point:
                r = random.choice(list)
            breakPoints.append(r)

        for p in breakPoints:
            l = []
            l.append(p)
            clusters[p] = l

        for i in clusters[point]:
            if i in breakPoints:
                # i += 1
                continue
            else:
                dist = {}
                for p in breakPoints:
                    # for clusterPoint in clusters[point]:

                    #d = optimized.calculateDist(tfidf[p], tfidf[i])
                    d = calculateDist(tfidf[p], tfidf[i])
                    if p in dist:
                        if dist[p] > d:
                            dist[p] = d
                    else:
                        dist[p] = d

                dist = sorted(dist.items(), key=operator.itemgetter(1))

                ##Check
                clusters[dist[0][0]].append(i)
                pointToClusterMap[i] = dist[0][0]
                print("Point " + str(i) + " went to cluster " + str(dist[0][0]))

                # i += 1
        clusters.pop(point)
        sse.pop(point)

        mean1 = calculateMean(clusters[breakPoints[0]])
        sse1 = calculateSSE(clusters[breakPoints[0]], mean1)
        sse[breakPoints[0]] = sse1
        clus1 = breakPoints[0]


        mean2 = calculateMean(clusters[breakPoints[1]])
        sse2 = calculateSSE(clusters[breakPoints[1]], mean2)
        sse[breakPoints[1]] = sse2
        clus2 = breakPoints[1]

        max = 0
        maxPoint = 0
        for k,v in sse.items():
            if v>max:
                max=v
                maxPoint=k
        breakCluster(maxPoint)
        return None


    for l in tfidf:
        global clusterIndex
        if i in intialPoints:
            i += 1
            continue
        else:
            dist = {}
            for point in intialPoints:
                # for clusterPoint in clusters[point]:

                #d = optimized.calculateDist(tfidf[point], tfidf[i])
                d = calculateDist(tfidf[point], tfidf[i])

                if point in dist:
                    if dist[point] > d:
                        dist[point] = d
                else:
                    dist[point] = d

            dist = sorted(dist.items(), key=operator.itemgetter(1))

            ##Check
            clusters[dist[0][0]].append(i)
            pointToClusterMap[i] = dist[0][0]
            print("Point " + str(i) + " went to cluster " + str(dist[0][0]))

            i += 1


    mean1 = calculateMean(clusters[intialPoints[0]])
    sse1 = calculateSSE(clusters[intialPoints[0]],mean1)
    sse[intialPoints[0]]=sse1
    clus1 = intialPoints[0]
    clusterIndex += 1

    mean2 = calculateMean(clusters[intialPoints[1]])
    sse2 = calculateSSE(clusters[intialPoints[1]],mean2)
    sse[intialPoints[1]]=sse2
    clus2 = intialPoints[1]
    clusterIndex += 1
    if sse1 > sse2:
        breakCluster(clus1)
    else:
        breakCluster(clus2)


    def distFromMean(mean, point):
        d=0
        for i in range(len(mean)):
            d+=(mean[i]-tfidf[point][i])**2
        d=d**0.5
        return d


    def calculateMedoid(mean, list):
        dist={}
        for point in list:
            d=distFromMean(mean,point)
            dist[point]=d
        dist = sorted(dist.items(), key=operator.itemgetter(1))

        return dist[0][0]


    def improveCluster():
        global clusters
        medoidList=[]
        for k,v in clusters.items():
            mean=calculateMean(v)
            medoid=calculateMedoid(mean,v)
            medoidList.append(medoid)
        i=0
        clusters = {}
        for point in medoidList:
            list = []
            list.append(point)
            clusters[point] = list
            pointToClusterMap[point] = point
        for l in tfidf:
            global clusterIndex
            if i in medoidList:
                i += 1
                continue
            else:
                dist = {}
                for point in medoidList:
                    # for clusterPoint in clusters[point]:

                    #d = optimized.calculateDist(tfidf[point], tfidf[i])
                    d = calculateDist(tfidf[point], tfidf[i])

                    if point in dist:
                        if dist[point] > d:
                            dist[point] = d
                    else:
                        dist[point] = d

                dist = sorted(dist.items(), key=operator.itemgetter(1))

                ##Check
                clusters[dist[0][0]].append(i)
                pointToClusterMap[i] = dist[0][0]
                print("Point " + str(i) + " went to cluster " + str(dist[0][0]))

                i += 1
        for k,v in clusters.items():
            mean=calculateMean(v)
            medoid = calculateMedoid(mean, v)
            if medoid in medoidList:
                continue
            improveCluster()
            return

        return


    improveCluster()
    print(len(pointToClusterMap))
    '''if numOfClusters==7:
        ans = open("ans1.dat", "+r")
        for i in range(len(pointToClusterMap)):
            kth = pointToClusterMap[i]
            j = 1
            for k, v in clusters.items():
                if kth == k:
                    break
                j += 1
            ans.write(str(j) + "\n")
    totalSSE=0
    for k,v in clusters.items():
        mean=calculateMean(clusters[k])
        totalSSE+=calculateSSE(clusters[k],mean)
    ssePerClusterIteration.append(totalSSE)
    sseVsCluster.append(numOfClusters)'''
    totalDatatSet+=1000
    totalTime1.append(time.time()-startTime)
    #numOfClusters+=2




proximityMatrix = []
i = 0
j = 0
numOfClusters=7
ssePerClusterIteration=[]
sseVsCluster=[]
totalTime2=[]
allx2=[]
totalDatatSet=1580
for x in range(8):
    startTime = time.time()

    tfidf=tfidfOri[:totalDatatSet][:]
    allx2.append(totalDatatSet)

    '''if pickleFile1.is_file():
        with open("pickleFile1.txt", "rb") as myFile:
            print("Loaing pickle!!")
            proximityMatrix = pickle.load(myFile)
    else:
        print("Creating pickle!!")
        print("Calculating proximity matrix!!")
        for i in range(len(tfidf)):
            list = []
            for j in range(len(tfidf)):
                if i == j:
                    d = 0
                else:
                    d = calulateDist(tfidf[i], tfidf[j])
                list.append(d)
            proximityMatrix.append(list)
        with open("pickleFile1.txt", "wb") as myFile:
            pickle.dump(proximityMatrix, myFile)'''
    print("Starting clustering!!")
    clusterIndex=0
    for i in range(2):

        intialPoints = []

        for i in range(2):
            intialPoints.append(randint(0,totalDatatSet-1))

        clusters = {}
        for point in intialPoints:
            list = []
            list.append(point)
            clusters[point] = list
            pointToClusterMap[point] = point



    i = 0

    sse = {}


    def calculateSSE(list, mean):
        l=[]
        ans=0
        for j in range(len(tfidf[0])):
            l.append(0)
            for i in range(len(list)):
                l[j]+=(mean[j]-tfidf[list[i]][j])**2


        for a in l:
            ans+=a
        a/=len(tfidf[0])
        return a


    def calculateMean(list):
        l = []
        ans = 0
        for j in range(len(tfidf[0])):
            l.append(0)
            for i in range(len(list)):
                l[j] +=  tfidf[list[i]][j]
        return l


    '''def calculateDist(p1, p2):
        d = 0
        for i in range(len(p1)):
            d+=(p1[i]-p2[i])**2
        d=d**0.5
        return d'''


    def breakCluster(point):
        global clusterIndex,clusters,numOfClusters
        if len(clusters) == numOfClusters:
            return None
        list = clusters[point]
        breakPoints = []
        for i in range(2):
            r = random.choice(list)
            if r == point:
                r = random.choice(list)
            breakPoints.append(r)

        for p in breakPoints:
            l = []
            l.append(p)
            clusters[p] = l

        for i in clusters[point]:
            if i in breakPoints:
                # i += 1
                continue
            else:
                dist = {}
                for p in breakPoints:
                    # for clusterPoint in clusters[point]:

                    d = optimized.calculateDist(tfidf[p], tfidf[i])
                    #d = calculateDist(tfidf[p], tfidf[i])
                    if p in dist:
                        if dist[p] > d:
                            dist[p] = d
                    else:
                        dist[p] = d

                dist = sorted(dist.items(), key=operator.itemgetter(1))

                ##Check
                clusters[dist[0][0]].append(i)
                pointToClusterMap[i] = dist[0][0]
                print("Point " + str(i) + " went to cluster " + str(dist[0][0]))

                # i += 1
        clusters.pop(point)
        sse.pop(point)

        mean1 = calculateMean(clusters[breakPoints[0]])
        sse1 = calculateSSE(clusters[breakPoints[0]], mean1)
        sse[breakPoints[0]] = sse1
        clus1 = breakPoints[0]


        mean2 = calculateMean(clusters[breakPoints[1]])
        sse2 = calculateSSE(clusters[breakPoints[1]], mean2)
        sse[breakPoints[1]] = sse2
        clus2 = breakPoints[1]

        max = 0
        maxPoint = 0
        for k,v in sse.items():
            if v>max:
                max=v
                maxPoint=k
        breakCluster(maxPoint)
        return None


    for l in tfidf:
        global clusterIndex
        if i in intialPoints:
            i += 1
            continue
        else:
            dist = {}
            for point in intialPoints:
                # for clusterPoint in clusters[point]:

                d = optimized.calculateDist(tfidf[point], tfidf[i])
                #d = calculateDist(tfidf[point], tfidf[i])

                if point in dist:
                    if dist[point] > d:
                        dist[point] = d
                else:
                    dist[point] = d

            dist = sorted(dist.items(), key=operator.itemgetter(1))

            ##Check
            clusters[dist[0][0]].append(i)
            pointToClusterMap[i] = dist[0][0]
            print("Point " + str(i) + " went to cluster " + str(dist[0][0]))

            i += 1


    mean1 = calculateMean(clusters[intialPoints[0]])
    sse1 = calculateSSE(clusters[intialPoints[0]],mean1)
    sse[intialPoints[0]]=sse1
    clus1 = intialPoints[0]
    clusterIndex += 1

    mean2 = calculateMean(clusters[intialPoints[1]])
    sse2 = calculateSSE(clusters[intialPoints[1]],mean2)
    sse[intialPoints[1]]=sse2
    clus2 = intialPoints[1]
    clusterIndex += 1
    if sse1 > sse2:
        breakCluster(clus1)
    else:
        breakCluster(clus2)


    def distFromMean(mean, point):
        d=0
        for i in range(len(mean)):
            d+=(mean[i]-tfidf[point][i])**2
        d=d**0.5
        return d


    def calculateMedoid(mean, list):
        dist={}
        for point in list:
            d=distFromMean(mean,point)
            dist[point]=d
        dist = sorted(dist.items(), key=operator.itemgetter(1))

        return dist[0][0]


    def improveCluster():
        global clusters
        medoidList=[]
        for k,v in clusters.items():
            mean=calculateMean(v)
            medoid=calculateMedoid(mean,v)
            medoidList.append(medoid)
        i=0
        clusters = {}
        for point in medoidList:
            list = []
            list.append(point)
            clusters[point] = list
            pointToClusterMap[point] = point
        for l in tfidf:
            global clusterIndex
            if i in medoidList:
                i += 1
                continue
            else:
                dist = {}
                for point in medoidList:
                    # for clusterPoint in clusters[point]:

                    d = optimized.calculateDist(tfidf[point], tfidf[i])
                    #d = calculateDist(tfidf[point], tfidf[i])

                    if point in dist:
                        if dist[point] > d:
                            dist[point] = d
                    else:
                        dist[point] = d

                dist = sorted(dist.items(), key=operator.itemgetter(1))

                ##Check
                clusters[dist[0][0]].append(i)
                pointToClusterMap[i] = dist[0][0]
                print("Point " + str(i) + " went to cluster " + str(dist[0][0]))

                i += 1
        for k,v in clusters.items():
            mean=calculateMean(v)
            medoid = calculateMedoid(mean, v)
            if medoid in medoidList:
                continue
            improveCluster()
            return

        return


    improveCluster()
    print(len(pointToClusterMap))
    '''if numOfClusters==7:
        ans = open("ans1.dat", "+r")
        for i in range(len(pointToClusterMap)):
            kth = pointToClusterMap[i]
            j = 1
            for k, v in clusters.items():
                if kth == k:
                    break
                j += 1
            ans.write(str(j) + "\n")
    totalSSE=0
    for k,v in clusters.items():
        mean=calculateMean(clusters[k])
        totalSSE+=calculateSSE(clusters[k],mean)
    ssePerClusterIteration.append(totalSSE)
    sseVsCluster.append(numOfClusters)'''
    totalDatatSet+=1000
    totalTime2.append(time.time()-startTime)
plt.plot(allx1,totalTime1)
plt.plot(allx2,totalTime2)

plt.ylabel("Time in seconds")
plt.xlabel("Datasize")
plt.legend(['Normal Python', 'Cython+OpenMP'], loc='upper left')
plt.show()
'''print(ssePerClusterIteration)
print(sseVsCluster)
plt.plot(sseVsCluster,ssePerClusterIteration)
plt.ylabel("SSE")
plt.xlabel("Num of clusters")
plt.show()
    for i in range(len(pointToClusterMap)):
        kth=pointToClusterMap[i]
        j=1
        for k,v in clusters.items():
            if kth==k:
                break
            j+=1
        ans.write(str(j)+"\n")
    # print(clusters)

endTime = time.time()
print("End time is:"+str(endTime))
print(endTime - startTime)'''