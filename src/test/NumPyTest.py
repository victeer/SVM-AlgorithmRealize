'''
Created on 2014-4-16

@author: weiwei
'''
import numpy as np


def func(i):
    return i*i+1


if __name__=='__main__':
    print (-1/(2*0.1**2))

#    X=np.array([[1,1],[1,2],[1,3]])
#    Y=np.array([[1,1],[2,2],[1,3]])
#    print Y[:,1]
##    print X.shape
##    print X
##    print X*X
##    print np.sum(X*X,axis=1)
##    print np.sum(np.dot(Y,X.T),axis=1)
##    print np.sum(np.dot(Y,X.T),axis=0)
#    X2=np.array([1,2,3])
#    print X2.shape
#    print np.dot(X2.reshape(-1,1),X)
    
#    print (X2+(-2)*np.dot(Y,X.T))    
#    print X2.T+(-2)*np.dot(Y,X.T)
#    print X2.reshape(1,-1)+(X2.reshape(-1,1)+(-2)*np.dot(Y,X.T))  
#    print  np.zeros((10,))
#    
#    print np.fromfunction(func, (10,))
#    a=np.array([1.0,1.0,2,2])
#    print np.sum(a)
#    a=a.reshape(2,2)
#    b=np.array([[3],[4]])
#    print a 
#    print b 
#    print a+b
#    dd=np.zeros((2,10))
#    print dd[:,1]
#    c=np.ones((5,))
#    
#    print c
#    c=np.ones((1,5))
#    d=np.ones((5,))
#    print c
#    print d
#    print c*d
#    print np.dot(c,d)
#    alphas=np.array([1, 2,3])
#    print alphas
#    Y=np.array([2, 4])
#    print Y
#    X=np.array([[2, 3 ,4],[1, 2, 3]])
#    print np.add(Y.reshape(2,1),X)
#    print alphas*X
    
#    print X.shape
#    print (alphas*Y).shape
#    print alphas*Y
#    print np.dot((alphas*Y),X)
#    print np.dot((alphas*Y).T,X)
#    print np.dot(X.T,alphas)
#    print np.dot(X.T,X)
#    print np.dot(X,X.T)
#    X1=np.sum(X*X, axis=1)
#    X2=X1.reshape(1,-1)
#    print X1.shape
#    print X2.shape
#    d=100
#    print c<d
#    f= np.array((1,))
#    print f
#    print c.shape
#    print a*a 
#    print np.dot(a,a)
#    print 2*(np.dot(a,a))
#    print np.sum(a*a,axis=1)
#    print np.dot(a,a)
#    print a*a
#    b = np.array((5, 6, 7, 8))
#    print b
#    
#    c = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])
#    print c
#    print c.shape
#    c.shape = 4,3
#    print c
#    c.shape=2,-1
#    print c 
#    d=a.reshape((2,2))
#    print d 
#    print a 
#    ff=np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]], dtype=np.float)

#    ff=np.arange(0,1,0.1)
#    ff=np.linspace(0, 1, 12)
#    ff=np.logspace(0, 2, 20)
#    print ff
#    s = "abcdefgh"
#    ff=np.fromstring(s, dtype=np.int8)
#    ff=np.fromstring(s, dtype=np.int16)
#    ff=np.fromstring(s, dtype=np.float)
#    print ff
#    a=np.arange(10)
#    a[:4]
#    print a[1:10:2]
#    b=np.array([[[1],[2],[3]], [[4],[5],[6]]])
#    print b.shape
#    print b[1:2]
#    a=np.mat('1 2 3; 4 5 3')
#    print (a*a.T).I
#    a = np.arange(0, 60, 10).reshape(-1, 1)
#    print a 