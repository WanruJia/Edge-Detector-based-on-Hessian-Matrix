import cv2
import xlwt
import numpy as np
import pandas as pd
import math

#Saving Brightness
image = cv2.imread("C:/DalMasterCourses/Medical Image Analysis/Xray Grades/program for X-ray images/APJointCentre/1-1R AP.jpg",1)
image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#image = cv2.resize(image,(2048,2048))
df=pd.DataFrame(image)
B = np.array(df).transpose()
#print(B)
#df.to_csv('20200418/Brightness.csv',index=False,header=False)

#Kernel Calculation
k = 9

x = []
y = []

for val in range(k):
    x.append([[-((k-1)/2-val)]*k])
    y.append([[-((k-1)/2-val)]*k])
x = np.array(x)
y = np.array(y).transpose()
#print(y)

X = np.zeros((6,k*k))
#print(x[0]**2)

for val in range(k):
    X[0][((val)*k):((val+1)*k)] = (x[val])**2
    X[1][((val)*k):((val+1)*k)] = (x[val])*y[val]
    X[2][((val)*k):((val+1)*k)] = (y[val])**2
    X[3][((val)*k):((val+1)*k)] = (x[val])
    X[4][((val)*k):((val+1)*k)] = (y[val])
    X[5][((val)*k):((val+1)*k)] = 1
#print(X)
X = X.transpose()
Kernel = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
#print(Kernel)

#EigenValues/Vectors Calculation
#print(df.shape[0])
Value1 = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1))
Value2 = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1))

Vector1 = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1,2))
Vector2 = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1,2))
#print(Vector2)

First_x = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1))
First_y = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1))

D = np.zeros((2,2))
for i in range(df.shape[0]-k+1):
    for j in range(df.shape[1]-k+1):
        Y = []
        for h in range(k):
            Y += B[j+h][i:(i+k)].tolist()
        Beta = np.dot(Kernel, Y)
        #print(Beta)
        D[0][0]=Beta[0]
        D[0][1]=Beta[1]/2
        D[1][0]=Beta[1]/2
        D[1][1]=Beta[2]
        Ev = np.linalg.eig(np.array(D))
        #print(Ev)
        if Ev[0][1] > Ev[0][0]:
            Value1[i][j] = Ev[0][1]
            Value2[i][j] = Ev[0][0]
            Vector1[i][j] = Ev[1][1]
            Vector2[i][j] = Ev[1][0]
        else:
            Value1[i][j] = Ev[0][0]
            Value2[i][j] = Ev[0][1]
            Vector1[i][j] = Ev[1][0]
            Vector2[i][j] = Ev[1][1]

        First_x[i][j] = Beta[3]
        First_y[i][j] = Beta[4]
#Diff = Value1 - Value2
#print(Y)
#print(Beta)
#value1_new=pd.DataFrame(Value1)
#value1_new.to_csv('20201012/Value1.csv',index=False,header=False)
#value2_new=pd.DataFrame(Value2)
#value2_new.to_csv('20201012/Value2.csv',index=False,header=False)
#vector1_new=pd.DataFrame(Vector1.reshape((df.shape[0]-k+1,(df.shape[1]-k+1)*2)))
#vector1_new.to_csv('20200502/Vector1.csv',index=False,header=False)
#vector2_new=pd.DataFrame(Vector2.reshape((df.shape[0]-k+1,(df.shape[1]-k+1)*2)))
#vector2_new.to_csv('20200502/Vector2.csv',index=False,header=False)
firstx_new = pd.DataFrame(First_x)
firstx_new.to_csv('20201212/FirstX.csv',index=False,header=False)
firsty_new = pd.DataFrame(First_y)
firsty_new.to_csv('20201212/FirstY.csv',index=False,header=False)
#print('Debug: 'Vector1[0])
#Score Function
w1 = 2.0
w2 = 0.2
threshold_link = -0.1

link = []

for i in range(df.shape[0]-k-k):
    for j in range(df.shape[1]-k-k):
        point1 = np.array([i,j])
        lambda1 = Value2[i][j]
        vx2 = Vector2[i][j]
        for h in range(i,i+k):
            for l in range(max((j-k),0),j+k):
                point2 = np.array([h,l])
                lambda2 = Value2[h][l]
                distanceyx = math.sqrt(sum(np.subtract(point1,point2)**2))
                vy2 = Vector2[h][l]
                dis = threshold_link-1
                if distanceyx != 0:
                    dis = lambda1*lambda2-w1*(abs(sum(np.multiply(np.subtract(point1,point2),vx2))+abs(sum(np.multiply(np.subtract(point1,point2),vy2))))*distanceyx**-0.5)-w2*distanceyx
                    if dis > threshold_link:
                        link.append(point1.tolist())
                        link.append(point2.tolist())
#print(link)
link_new=pd.DataFrame(link)
link_new.to_csv("C:/DalMasterCourses/Softwares for Hongyang's Group/LinkPoints-01.csv",index=False,header=False)
link_trans=np.array(link_new)
link_trans[:,[0, 1]] = link_trans[:,[1, 0]]
#Draw the images
times = 5
image_new = np.zeros((df.shape[1]*times,df.shape[0]*times,1), dtype="uint8")
image_new = cv2.resize(image_new,(df.shape[1]*times,df.shape[0]*times))
image_origin = cv2.resize(image,(df.shape[1]*times,df.shape[0]*times))


for i in range(int(len(link_trans)/2)):
	image_pure = cv2.line(image_new,tuple(link_trans[i*2]*times),tuple(link_trans[i*2+1]*times),255, thickness=1)
cv2.imshow('Edge',image_pure)
cv2.waitKey(0)
cv2.imwrite("20201212/Edgeonly"+str(times)+'times 01'+ str(threshold_link) +' kernel '+str(k)+'('+ str(w1) +str(w2) +').jpg', image_new)

outputImage = image
outputImage = cv2.resize(outputImage,(df.shape[1]*times,df.shape[0]*times))
for i in range(int(len(link_trans)/2)):
    image_superimpose = cv2.line(outputImage,tuple((link_trans[i*2]*times)),tuple((link_trans[i*2+1]*times)),(255, 0, 0), thickness=1)

cv2.imshow('Edge', image_superimpose)
cv2.waitKey(0)
cv2.imwrite("20201212/Edge superimpose oringin"+str(times)+'times 01'+ str(threshold_link) +' kernel '+str(k) +'('+ str(w1) +str(w2) +').jpg',image_superimpose)


#print(image_new, image_superimpose, image_pure)
#numpy_horizontal = np.hstack((image_new, image_superimpose, image_pure))
#numpy_horizontal_concat = np.concatenate((image_origin, image_superimpose, image_pure), axis=1)
#cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
#cv2.waitKey()
#cv2.imwrite('20200529/Edge superimpose oringin'+str(times)+'times 06 9by9'+ str(threshold_link) +'('+ str(w1) +str(w2) +').jpg',numpy_horizontal_concat)

cv2.destroyAllWindows()
