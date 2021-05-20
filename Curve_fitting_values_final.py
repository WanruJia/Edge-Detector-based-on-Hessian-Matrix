import cv2
import xlwt
import numpy as np
import pandas as pd
from numpy import linalg as LA
import math
from scipy.integrate import dblquad

#Saving Brightness
image = cv2.imread("C:/DalMasterCourses/Medical Image Analysis/Xray Grades/program for X-ray images/APJointCentre/1-1R AP.jpg",1)
image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#image_gray = cv2.resize(image_gray,(2048,2048))
df=pd.DataFrame(image_gray)
B = np.array(df).transpose()
#print(B)
#df.to_csv('20201018/Brightness-01-whole.csv',index=False,header=False)

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
    X[0][((val)*k):((val+1)*k)]=(x[val])**2
    X[1][((val)*k):((val+1)*k)]=(x[val])*y[val]
    X[2][((val)*k):((val+1)*k)]=(y[val])**2
    X[3][((val)*k):((val+1)*k)]=(x[val])
    X[4][((val)*k):((val+1)*k)]=(y[val])
    X[5][((val)*k):((val+1)*k)]=1
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

H = np.zeros((df.shape[0]-k+1,df.shape[1]-k+1,4))
#print(Vector2)

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
        H[i][j] = D.flatten()
Diff = Value1 - Value2
#print("Debug:" & Beta)
#value1_new=pd.DataFrame(Value1)
#value1_new.to_csv('20201012/Value1.csv',index=False,header=False)
#value2_new=pd.DataFrame(Value2)
#value2_new.to_csv('20201012/Value2.csv',index=False,header=False)
#vector1_new=pd.DataFrame(Vector1.reshape((df.shape[0]-k+1,(df.shape[1]-k+1)*2)))
#vector1_new.to_csv('20200502/Vector1.csv',index=False,header=False)
#vector2_new=pd.DataFrame(Vector2.reshape((df.shape[0]-k+1,(df.shape[1]-k+1)*2)))
#vector2_new.to_csv('20200502/Vector2.csv',index=False,header=False)
# Objective Function

J = np.array([[0,1],[-1,0]])
vlink = []

# Set the beginning values for Hessian Matrices, K and v
#result = np.where(Vector2[:,1] == np.amin(Vector2[:,1]))
#print('Returned tuple of arrays :', result)
#print('List of Indices of minimum element :', result[0])

def hessian(x,y):
    frx, inx = math.modf(x)
    fry, iny = math.modf(y)
    inx = int(inx)
    iny = int(iny)
    H1 = np.resize(H[inx,iny],(2,2))
    H2 = np.resize(H[inx+1,iny],(2,2))
    H3 = np.resize(H[inx,iny+1],(2,2))
    H4 = np.resize(H[inx+1,iny+1],(2,2))
    H_xy = frx*fry*H4 + fry *(1-frx)*H3 + frx*(1-fry)*H2 + (1-frx)*(1-fry)*H1
    return H_xy

def objectfun(x,y,start0,start1):
    vloc=(start0+x,start1+y)
    dloc = np.array([x,y])
    H_local = hessian(vloc[0],vloc[1])
    K_local = np.dot(np.dot(-J,np.resize(H_local,(2,2))),J)
    object_local = np.dot(np.dot(dloc.transpose(),K_local),dloc)
    return object_local

outputImage_origin = cv2.resize(image[4:df.shape[0]-k//2,4:df.shape[1]-k//2],((df.shape[0]-k+1)*10,(df.shape[1]-k+1)*10))
edge_only = cv2.imread("C:/DalMasterCourses/Medical Image Analysis/Xray Grades/program for X-ray images/20200502/twobytwo eigenvalue.jpg",1)
edge_only = cv2.resize(edge_only,(1200,1200))


# Set tunning parameter and steps
starts = [(38, 60)]
point_num = 1
windowsize = 11
steps = 9
upper = df.shape[0] - k - 1

for start in starts:
    for loop_num in range(point_num):
        vstart = (start[0]-loop_num,start[1])
        new = []
        v0 = vstart
        index = 0
        direction = np.array([0,0])
        while int(v0[0])<upper and int(v0[1])<upper and int(v0[1])>-int((direction*steps)[1]) and int(v0[0])>-int((direction*steps)[0]):
            circlepoints = {}
            for i in range(-windowsize//2,windowsize//2+1):
                for j in range(-windowsize,1,1):
                    v=(max(min(v0[0]+i,upper),0),max(min(v0[1]+j+1,upper),0))
                    d = np.array([-i,-j])
                    V2 = Value2[int(v[0]),int(v[1])]
                    circlepoints[V2]=v

            v = circlepoints[min(circlepoints)]

            if index == 0:
                H_sum = np.zeros((2,2))
                count = 0
                dis = math.sqrt((v[0]-v0[0])**2+(v[1]-v0[1])**2)
                if dis == 0:
                    index += 1
                    continue
                for x in range(int(dis)):
                    H_sum += hessian(v0[0]+x/dis*(v[0]-v0[0]),v0[1]+x/dis*(v[1]-v0[1]))
                    count += 1
                K_sum = np.dot(np.dot(-J,np.resize(H_sum,(2,2))),J)
                dd = np.array([(v[0]-v0[0])/dis,(v[1]-v0[1])/dis])
                score = np.dot(np.dot(dd.transpose(),K_sum),dd)/count
                print(vstart, score)
            index += 1

            direction = Vector2[int(v[0])][int(v[1])]
            v0 = v0 - direction*steps
            #print(v[1],v[0])
            new.append(v)
        save_1 = new[0]

        for i in range(int(len(new)-1)):
            image_new_origin = cv2.line(outputImage_origin,(int(new[i][1]*10),int(new[i][0]*10)),(int(new[i+1][1]*10),int(new[i+1][0]*10)),(0, 255, 0), thickness=3)
            edge_only = cv2.line(edge_only,(int(new[i][1]*10),int(new[i][0]*10)),(int(new[i+1][1]*10),int(new[i+1][0]*10)),(0, 0, 255), thickness=3)

        vstart = (start[0]-loop_num,start[1])
        new = []
        v0 = vstart
        index = 0

        while int(v0[0])<upper and int(v0[1])<upper and int(v0[1])>0 and int(v0[0])>0:
            circlepoints = {}
            for i in range(-windowsize//2,windowsize//2+1):
                for j in range(windowsize):
                    v=(max(min(v0[0]+i,upper),0),max(min(v0[1]+j+1,upper),0))
                    d = np.array([i,j])
                    V2 = Value2[int(v[0])][int(v[1])]
                    circlepoints[V2]=v

            if index == 0:
                H_sum = np.zeros((2,2))
                count = 0
                dis = math.sqrt((v[0]-v0[0])**2+(v[1]-v0[1])**2)
                for x in range(int(dis)):
                    H_sum += hessian(v0[0]+x/dis*(v[0]-v0[0]),v0[1]+x/dis*(v[1]-v0[1]))
                    count += 1
                K_sum = np.dot(np.dot(-J,np.resize(H_sum,(2,2))),J)
                dd = np.array([(v[0]-v0[0])/dis,(v[1]-v0[1])/dis])
                score = np.dot(np.dot(dd.transpose(),K_sum),dd)/count
                print(vstart, score)
            index += 1

            v = circlepoints[min(circlepoints)]
            direction = Vector2[int(v[0])][int(v[1])]
            v0 = v0 + direction*steps
            #print(v[1],v[0])
            new.append(v)
        save_2 = new[0]

        for i in range(int(len(new)-1)):
            image_new_origin = cv2.line(outputImage_origin,(int(new[i][1]*10),int(new[i][0]*10)),(int(new[i+1][1]*10),int(new[i+1][0]*10)),(0, 255, 0), thickness=3)
            edge_only = cv2.line(edge_only,(int(new[i][1]*10),int(new[i][0]*10)),(int(new[i+1][1]*10),int(new[i+1][0]*10)),(0, 0, 255), thickness=3)
    image_new_origin = cv2.line(outputImage_origin,(int(save_1[1]*10),int(save_1[0]*10)),(int(save_2[1]*10),int(save_2[0]*10)),(0, 255, 0), thickness=3)
    edge_only = cv2.line(edge_only,(int(save_1[1]*10),int(save_1[0]*10)),(int(save_2[1]*10),int(save_2[0]*10)),(0, 0, 255), thickness=3)


cv2.imshow('Curve', image_new_origin)
cv2.waitKey(0)
cv2.imwrite('20201122/Method 3 01 '+str(windowsize)+str(starts)+str(steps)+str(point_num)+' starting points superimpose oringin new.jpg',image_new_origin)

cv2.imshow('Edge', edge_only)
cv2.waitKey(0)
cv2.imwrite('20201122/Method 3 01 '+str(windowsize)+str(starts)+str(steps)+str(point_num)+' vector plots.jpg',edge_only)
