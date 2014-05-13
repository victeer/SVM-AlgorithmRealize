# -*- coding: utf-8 -*-
'''
Created on 2014-3-4

@author: weiwei
'''
import math;
import random;


class svm(object): 
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def linearK(self):
        ''' 
        calculate linear self.K for train function 
        '''
        self.K =[[0 for row in range(0,self.m)] for column in range(0,self.m)];
        for i in range(0,self.m):
            for j in range(i,self.m):
                tmpK=0;
                for  t in range(0,self.n):
                    tmpK+=self.X[i][t] * self.X[j][t];
                self.K[i][j]=self.K[j][i]=tmpK;
            
    def gaussianK(self,Sigma):
        '''
        calculate gaussian self.K for training
        K(x,z)=exp(-||x-z||**2/(2*sigma**2))
        '''  
        self.K =[[0 for row in range(0,self.m)] for column in range(0,self.m)];
        for i in range(0,self.m):
            for j in range(i,self.m):
                tmpK=0;
                for  t in range(0,self.n):
                    tmpK+=(self.X[i][t] - self.X[j][t])**2;
                tmpK=math.exp(-tmpK/(2*Sigma**2));    
                self.K[i][j]=self.K[j][i]=tmpK;

            

    def getE(self,i):
        '''
        '''
        e=0;
        e+=self.b;
        e-=self.Y[i];

        for t in range(0,self.m):
            e+=self.alphas[t]*self.Y[t]*self.K[t][i];
        return e;
    
    def getY(self,Y,first):
        '''
        one2one 和one2rest 两者的getY的方法都一致，所以就用同一个函数了。
        '''
        newY=[-1 for row in range(0,len(Y))];
        for i in range(0,len(Y)):
            if Y[i]==first:
                newY[i]=1;
        return newY
    
        
                
    def train(self,X,Y,first, second,kernelFunction,Sigma=0.1,C=1,tol=1e-3,max_passes=5):
        '''
        first is the first class, second is the second class.
        X  is a group of student. I like it.
        '''
        self.X=X;
        self.Y=Y

            
        self.m=len(X);
        self.n=len(X[0]);
        
        #Variables
        self.alphas=[0 for row in range(0,self.m)];
        self.b=0;
        self.E=[0 for row in range(0,self.m)];
        passes=0;
        eta=0;
        L=0;
        H=0;
        
#        for i in range(len(self.Y)):
#            if self.Y[i]==0:
#                self.Y[i]=-1;
        
        # Pre-compute the Kernel Matrix since our dataset is small
        #(in practice, optimized SVM packages that handle large datasets
        #  gracefully will _not_ do this)
        #
        if (kernelFunction=='linearKernel'):
            self.linearK();
        elif(kernelFunction=='gaussianKernel'):
            self.gaussianK(Sigma);
        else:
            print 'no kernel support!' ;
            return -1;
        
        print "begin training!"
        dots=12;
        while passes<max_passes :
            
            num_changed_alphas=0;
            
            for i in range(0,self.m):
                self.E[i]=self.getE(i);
                #print "E[i]",self.E[i]
                if (self.Y[i]*self.E[i]<-tol and self.alphas[i]<C) or (self.Y[i]*self.E[i] > tol and self.alphas[i]>0):
                    # error random cannot find.
                    j=random.randint(0,self.m-1);  
                    
                    while j==i:#make sure i \neq j
                        j=random.randint(0,self.m-1);
                    
                    self.E[j]=self.getE(j);
                    alpha_i_old=self.alphas[i];
                    alpha_j_old=self.alphas[j];
                    if self.Y[i]==self.Y[j]:
                        L= max(0,self.alphas[j]+self.alphas[i]-C);
                        H=min(C,self.alphas[j]+self.alphas[i]);
                    else:
                        L=max(0,self.alphas[j]-self.alphas[i]);
                        H=min(C,C+self.alphas[j]-self.alphas[i]);
                    
                    if L==H :
                        continue;
                    
                    eta=2*self.K[i][j]-self.K[i][i]-self.K[j][j];
                    if eta>=0:
                        continue;
                     
                    self.alphas[j]=self.alphas[j]-(self.Y[j]*(self.E[i]-self.E[j]))/eta;
                    # Clip
                    self.alphas[j]=min(H,self.alphas[j]);
                    self.alphas[j]=max(L,self.alphas[j]);
                     
                    if abs(self.alphas[j]-alpha_j_old)<tol :
                        self.alphas[j]=alpha_j_old;
                        continue;
                     
                    # Determine value for alpha i  
                    self.alphas[i] = self.alphas[i] + self.Y[i]*self.Y[j]*(alpha_j_old - self.alphas[j]);
                    #print"alphas[%d]: %f alphas[%d]: %f"  %(i,self.alphas[i],j,self.alphas[j])
                    #Compute b1 and b2
                    b1 = self.b - self.E[i]- self.Y[i] * (self.alphas[i] - alpha_i_old) *  self.K[i][i] - self.Y[j] * (self.alphas[j] - alpha_j_old) *  self.K[i][j];
                    b2 = self.b - self.E[j] \
                    - self.Y[i] * (self.alphas[i] - alpha_i_old) *  self.K[i][j]\
                    - self.Y[j] * (self.alphas[j] - alpha_j_old) *  self.K[j][j];

                    # Compute self.b 
                    if (0 < self.alphas[i] and self.alphas[i] < C):
                        self.b = b1;
                    elif (0 < self.alphas[j] and self.alphas[j] < C):
                        self.b = b2;
                    else:
                        self.b = (b1+b2)/2;
        
                    num_changed_alphas = num_changed_alphas + 1;
            
            if (num_changed_alphas == 0):
                passes = passes + 1;
            else:
                passes = 0;

            print ".",
            dots = dots + 1;
            if dots > 78:
                dots = 0;
                print "\n",;
        
        print ' Done! \n\n';

        #Save the model
        #print "save model:"
#        print self.alphas
        model=Model();
        for i in range(0,len(self.alphas)):
            if(self.alphas[i]>0):
#                print '第',i,'号',self.X[i],self.Y[i],self.alphas[i]
                model.X.append(self.X[i]);
                model.Y.append(self.Y[i]);
                model.alphas.append(self.alphas[i]);
        model.b=self.b;
        model.kernelFunction=kernelFunction;
        model.w=self.getW();
        model.Sigma=Sigma;
        model.first=first;
        model.second=second;
        return model;          
        
    def getW(self):
        '''
        get Weight vector after training end
        '''  
#        print 'n:',self.n  
        w=[0 for row in range(0,self.n)];
        for j in range(0,self.n):
            ww=0;
            for i in range(self.m):
                ww+=self.alphas[i]*self.Y[i]*self.X[i][j];
            w[j]=ww;
        return w
        
    def predict(self,model,X):
        m=len(X);
        n=len(X[0]);
        
        p=[0 for row in range(m)];
        pred=[model.second for row in range(m)];
        
        if (model.kernelFunction=='linearKernel'):
            for i in range(0,m):
                tmpP=0;
                for j in range(0,n):
                    tmpP+=X[i][j]*model.w[j];
                tmpP+=model.b;
                p[i]=tmpP;
                if(tmpP>=0):
                    pred[i]=model.first;    
        elif(model.kernelFunction=='gaussianKernel'):
            for i in range(0,m):
                prediction=0;
                for j in range(0,len(model.X)):
                    tmp=0;
                    #get ||x-z||**2
                    for t in range(0,n):
                        tmp+=(X[i][t] - model.X[j][t])**2;
                        
                    tmp=math.exp(-tmp/(2*model.Sigma**2));
                    prediction=prediction+model.alphas[j]*model.Y[j]*tmp;
                    
                prediction+=model.b;
                p[i]=prediction;
                if(prediction>=0):
                    pred[i]=model.first;
        else:
            print 'no kernel support!' ;
            return -1;
        return (pred,p);
    def predictPricision(self,oriY,preY):
        '''
        calculate the precision rate according to the compare of oriY and preY
        '''
        right=0
        for i in range(len(oriY)):
            if oriY[i]==preY[i]:
                right+=1
        print right, len(oriY),len(preY)       
        print "correct rate: ",float(right)/len(preY); 
    
    def printPreAndOriCompare(self,oriY,preY,out_name):
        file_out=open(out_name,'w');
        file_out.write('original Y \tpredict Y\n')
        for i in range(len(preY)):
            file_out.write(str(oriY[i])+'\t'+str(preY[i])+'\n')
            
        file_out.close()
    
    def multiClassPredict(self,models,X):
        '''
        思路：遍历所有的模型，最后得到每个模型下，对于X这样的数据的分类结果。最后统计X（i）的在每一个类下面的分类器投票结果，
        最后输出X(i)的投票最多的一个分类。
        （*如果有多个分类的数值相同，优先输出第一个这个数值的分类。以后可以考虑输出 多个分类。*）
        '''
        classJudgeArray=[]
        scoreJudgeArray=[]
        for model in models:
            predict,pScore=self.predict(model,X);
            classJudgeArray.append(predict)
            if model.second==-1:
                scoreJudgeArray.append(pScore)
        #begin to judge
        classArray=[0 for row in range(0,len(classJudgeArray[0]))]
        if len(scoreJudgeArray)==0:#means 1v1 is here.
            for i in range(0,len(classJudgeArray[0])):# for each example judge which class it belongs to.
                classDecision={}
                for j in range(0,len(classJudgeArray)):
                    className=classJudgeArray[j][i]
                    if classDecision.has_key(className):#className==-1:pass elif
                        count=classDecision.get(className);
                        count=count+1;
                        classDecision[className]=count;
                    else:
                        classDecision[className]=1;
                max=0;
                getClass=None;
                for key in classDecision:
                    if classDecision[key]>max:
                        max=classDecision[key];
                        getClass=key;
                classArray[i]=getClass; 
        else:#1vRest
            for i in range(0,len(classJudgeArray[0])):
                classDecision={}
                for j in range(0,len(classJudgeArray)):
                    className=classJudgeArray[j][i]
                    if className==-1:
                        pass
                    else:
                        classDecision[className]=j;#record the coordinate of class
                if len(classDecision)!=0:
                    getClass=None;
                    max=0.0
                    for key in classDecision:
                        pos=classDecision[key]
                        score=scoreJudgeArray[pos][i];
                        if max<score:
                            max=score;
                            getClass=key;
                    classArray[i]=getClass;
                else:
                    classArray[i]=-1;             
        return classArray;
        
        #begin to judge.
        
    def storeModels(self,models,modelsFile):
        '''
        store model to file, so that we can get the model from file later.
        '''
        F=open(modelsFile,'w')
        import pickle
        pickle.dump(models, F)
        F.close();
    
    def  readModels(self,modelsFile):
        '''
        read model from a file.
        '''
        F=open(modelsFile)
        import pickle
        models=pickle.load(F)
        return models;
        
        
    def multiClassOne2Rest(self,dataset,sigma=0.1,C=1,kernelFunction='linearKernel',tol=1e-3,max_passes=5):
        '''
        Training function.
        思路：首先main函数把数据读入内存中，
        将每一个类别的按照类别名称用一个dict将类别和该类别下的所有数据都对应起来，
        在这个函数中运用迭代的方法将每一个类别和其他类别输入train函数中，
        将返回的模型存储起来，或者以二进制的方式输出到文件，最终得到N个不同的分类器；
        供下一步预测的时候用。
        ''' 
        models=[]
        for key in dataset.keys():
            first=key;
            second=-1#means that the rest otherwise another class.
            X=list(dataset[key])
            firstSize=len(X);
            for otherkey in dataset.keys():
                if otherkey!=key:
                    '''
                    ##here i need to rejudge it. if the append work.
                    '''
                    X.extend(dataset[otherkey])
            Y=[-1 for row in range(0,len(X))]
            ####批量赋值 does it work?
            Y[0:firstSize]=[1 for row in range(0,firstSize)];
            model=self.train(X,Y,first, second,Sigma=sigma,C=C,kernelFunction=kernelFunction,tol=tol,max_passes=max_passes)
            models.append(model)

        return models
    
    
    def printObject(self,X):
        for  x in X:
            print x;
        
    def multiClassOne2One(self,dataset,sigma=0.1,C=1,kernelFunction='linearKernel',tol=1e-3,max_passes=5):
        '''
        思路大体与one2rest一致，区别在在对输入数据的迭代中，
        本方法每次迭代对train函数的输入是两个不同的类别的数据，
        而最终的结果是C(n,2)个分类器
        '''
        models=[]
        
        for i, key in enumerate(dataset):#this method is ok.
            first=key;
            X=list(dataset[key])
            firstSize=len(X);
            for  j, otherkey in enumerate(dataset):
                if j>i:
                    '''
                    ##here i need to rejudge it. if the append work.
                    '''
                    second=otherkey;
                    X.extend(dataset[otherkey])
                    Y=[-1 for row in range(0,len(X))]
                    ####批量赋值 does it work?
                    Y[0:firstSize]=[1 for row in range(0,firstSize)];
                    model=self.train(X,Y,first, second,Sigma=sigma,C=C,kernelFunction=kernelFunction,tol=tol,max_passes=max_passes)
                    models.append(model)
        return models
            
    def readTrainData(self,datasetPath):
        '''
            思路如下：首先读入数据，解析成class- 数据对的形式，
            然后，将同一个class下的数据用dict存储到同一个key下面
        '''
        dataset={}
        with open(datasetPath,'r') as f:
            for line in f:
                tmp=line.strip().split('\t')
                className=tmp[0].strip()
                if dataset.has_key(className):
                    dataset[className].append(map(float,tmp[1:]))
                else:
                    dataset[className]=[]
                    dataset[className].append(map(float,tmp[1:]))
        return dataset;
    
    def outputData(self,dataset):
        '''
            对每一个字典，打印其对应的值
            main use is outputing the trainning data getting from the readTrainData() function.
        '''
        for key in dataset:
            print key;
            for array in dataset[key]:
                print array;
    def getTestData(self,dataset):
        '''
        从字典序的dataset中获得测试数据和对应的类别标签
        '''
        X=[]
        Y=[]
        for key in dataset.keys():
            lenY=len(Y)
            lenX=len(dataset[key])
            X.extend(dataset[key])
            Y[lenY:lenY+lenX]=[key for row in range(0,lenX)]    
        return (X,Y)  
      
class Model(object):
    '''
    this model is create for storing the training result, including the support vector X and corresponding Y, 
    and weight for each feature, also alphas generated from optimal process, b, and the kernelFunction of the model.
    '''
    def __init__(self):
        self.X=[]
        self.Y=[]
        self.w=[];
        self.alphas=[];
        self.b=None;
        self.kernelFunction=None
        self.Sigma=None;
        self.first=None;
        self.second=None;
        
    def __str__(self, *args, **kwargs):
        s="model:"
        s+="\nX:\n"
        for  i in range (len(self.X)):
            for j in range(len(self.X[0])):
                s+=str(self.X[i][j])+'\t';
            s+='\n';
        s+="Y:\n"
        for  i in range (len(self.Y)):
            s+=str(self.Y[i])+'\n';
        s+="w:\n";
        for  i in range (len(self.w)):
            s+=str(self.w[i])+'\n';
        s+="alphas:\n";
        for  i in range (len(self.alphas)):
            s+=str(self.alphas[i])+'\n';
        s+="b:"+str(self.b)
        s+="\nkernel:"+str(self.kernelFunction)
        s+="\nsigma:"+str(self.Sigma)
        s+="\n first class:"+str(self.first)
        s+="\n second class:"+str(self.second)
        return s
        
if __name__=="__main__":
    svm=svm()
    dataset=svm.readTrainData("D:\\Project\\Java\\helloWorld\\svmData\\smallData\\toy.txt")
    models=svm.multiClassOne2One(dataset=dataset,kernelFunction='gaussianKernel')
#    svm.storeModels(models, 'D:/Project/Java/svm/model/models.txt')
#    models1=svm.readModels('D:/Project/Java/svm/model/models.txt');
    
    X=[]
    Y=[]
    for key in dataset.keys():
        lenY=len(Y)
        lenX=len(dataset[key])
        X.extend(dataset[key])
        Y[lenY:lenY+lenX]=[key for row in range(0,lenX)]
    preY=svm.multiClassPredict(models, X)
    svm.predictPricision(Y, preY);
#    svm.printPreAndOriCompare(Y, preY, 'D:/Project/Java/svm/result/precision.txt')
        
