import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from filtrage_particulaire import *

N = 20
Nb = 10
lam = 5
C1 = 900
C2 = 900

SEQUENCE = "PATH/TO/SEQUENCE"
filenames = os.listdir(SEQUENCE)
T = len(filenames)

im = lecture_image()
z = selectionner_zone()
plt.close()
hgx = z[0]
hgy = z[1]
longueur = z[2]
largeur = z[3]

x = np.random.normal(z[0],np.sqrt(C1),N)
y = np.random.normal(z[1],np.sqrt(C2),N)
part = np.array([[x[i],y[i]] for i in range(N)])
X = np.zeros((N,2))
for n in range(N):
    X[n] = [x[n],y[n]]
w = 1/N*np.ones((N,))
w = w/np.sum(w)
Xw = np.sum([X[:,0]*w,X[:,1]*w],axis=1)
href = calcul_histogramme(im[0],z,Nb)[2]

for t in range(1,T):
    D = []
    wt = np.zeros((N,))
    im = Image.open((str(SEQUENCE)+str(filenames[t])))
    print((str(SEQUENCE)+str(filenames[t])))
    x0 = X[(t-1)*N:t*N,0]
    x1 = X[(t-1)*N:t*N,1]
    Xt_1x = np.random.choice(list(x0),N,list(w[(t-1)*N:t*N]))
    Xt_1y = np.random.choice(list(x1),N,list(w[(t-1)*N:t*N]))
    Xt_1 = np.zeros((N,2))
    for n in range(N):
        Xt_1[n] = [Xt_1x[n],Xt_1y[n]]
    Xt = Xt_1 + np.random.multivariate_normal([0,0],[[C1,0],[0,C2]],N)
    plt.scatter(Xt[:,0],Xt[:,1],marker='X',color='blue') 
    for p in range(N):
        h = calcul_histogramme(im,[Xt[p][0],Xt[p][1],longueur,largeur],Nb)[2]
        s = 0
        for i in range(1,Nb+1):
            s += np.sqrt(h[i]*href[i])
        D.append(np.sqrt(1-s))
    wt = norm.pdf(np.exp(-lam*np.array(D)**2))
    wt = wt/np.sum(wt)
    X = np.vstack([X,Xt])
    w = np.vstack([w,wt])
    Xw = np.sum([Xt[:,0]*wt,Xt[:,1]*wt],axis=1)
    plt.scatter(Xw[0],Xw[1],marker='X',color='red')
    rect = ptch.Rectangle((Xw[0],Xw[1]),longueur,largeur,linewidth=1,edgecolor='red',facecolor='None')  
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)

plt.imshow(im)
plt.show()

