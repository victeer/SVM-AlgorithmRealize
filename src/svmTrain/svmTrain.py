# -*- coding: utf-8 -*-
'''
Created on 2014-3-4

@author: weiwei
'''
import math;
import random;
import numpy as np;

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
        self.K=np.dot(self.X,self.X.T)
        print self.K.shape
#        self.K =[[0 for row in range(0,self.m)] for column in range(0,self.m)];
#        for i in range(0,self.m):
#            for j in range(i,self.m):
#                tmpK=0;
#                for  t in range(0,self.n):
#                    tmpK+=self.X[i][t] * self.X[j][t];
#                self.K[i][j]=self.K[j][i]=tmpK;
    def writeGaussianK(self):        
        file_out=open('D:\\Project\\Java\\svm\\result\\gaussian.txt','w');
        print self.K
        for i in range(len(self.K)):
            for j in range(len(self.K[0])):
                file_out.write(str(self.K[i][j])+'\t')
            file_out.write('\n')
            
        file_out.close()
        
    def gaussianK(self,Sigma):
        '''
        calculate gaussian self.K for training
        K(x,z)=exp(-||x-z||**2/(2*sigma**2))
        '''  
        X2=np.sum(self.X*self.X,axis=1)
        self.K=X2.reshape(-1,1)+X2.reshape(1,-1)+(-2)*np.dot(self.X,self.X.T)
        
        
        #self.K=np.exp(self.K*(-1/(2*Sigma**2)))
        self.K=np.exp(self.K*(-Sigma))
#        self.writeGaussianK()
        
        
#        self.K =[[0 for row in range(0,self.m)] for column in range(0,self.m)];
#        for i in range(0,self.m):
#            for j in range(i,self.m):
#                tmpK=0;
#                for  t in range(0,self.n):
#                    tmpK+=(self.X[i][t] - self.X[j][t])**2;
#                tmpK=math.exp(-tmpK/(2*Sigma**2));    
#                self.K[i][j]=self.K[j][i]=tmpK;

            

    def getE(self,i):
        '''
        '''
        e=self.b-self.Y[i]+np.sum(self.alphas*self.Y*self.K[:,i])
#        for t in range(0,self.m):
#            e+=self.alphas[t]*self.Y[t]*self.K[t][i];
        return e;
    def getY(self,Y,first):
        '''
        one2one 和one2rest 两者的getY的方法都一致，所以就用同一个函数了。
        '''
        newY=[-1 for row in range(0,len(Y))];
        second=None
        for i in range(0,len(Y)):
            if Y[i]==first:
                newY[i]=1;
            elif second==None:
                second=Y[i];
        return (newY,first,second)
    
        
                
    def train(self,X,Y,modifyY=False,first=None, second=None,Sigma=0.1,C=1,kernelFunction='linearKernel',tol=1e-3,max_passes=5):
        '''
        first is the first class, second is the second class.这两个用来给出将来在模型中标明的第一类和第二类

        X  is a group of student. I like it.
        '''
        self.X=np.array(X);
        self.Y=Y
        if modifyY:
            self.Y,first,second=self.getY(Y,Y[0])
        
        self.Y=np.array(self.Y)
        self.m=len(X);
        self.n=len(X[0]);
        
        #Variables
        self.alphas=np.zeros((self.m,))#[0 for row in range(0,self.m)];
        self.b=np.zeros((1,))
        self.E=np.zeros((self.m,))#[0 for row in range(0,self.m)];
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
                self.E[i]=self.b-self.Y[i]+np.sum(self.alphas*self.Y*self.K[:,i])
#                self.E[i]=self.getE(i);
                #print "E[i]",self.E[i]
                if (self.Y[i]*self.E[i]<-tol and self.alphas[i]<C) or (self.Y[i]*self.E[i] > tol and self.alphas[i]>0):
                    # error random cannot find.
                    j=random.randint(0,self.m-1);  
                    
                    while j==i:#make sure i \neq j
                        j=random.randint(0,self.m-1);
                    
                    self.E[j]=self.b-self.Y[j]+np.sum(self.alphas*self.Y*self.K[:,j]);
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
        model.X=np.array(model.X)
        model.Y=np.array(model.Y)
        model.alphas=np.array(model.alphas)
        model.b=self.b;
        model.kernelFunction=kernelFunction;
        model.w=self.getW();
        model.Sigma=Sigma;
        model.first=first;
        model.second=second;
        print 'size of support vector:',len(model.alphas)
        return model;          
        
    def getW(self):
        '''
        get Weight vector after training end  列向量
        '''  
#        print 'n:',self.n  
        return np.dot((self.alphas*self.Y),self.X)
#        w=[0 for row in range(0,self.n)];
#        for j in range(0,self.n):
#            ww=0;
#            for i in range(self.m):
#                ww+=self.alphas[i]*self.Y[i]*self.X[i][j];
#            w[j]=ww;
#        return w
        
    def predict(self,model,X):
        X=np.array(X)
        m=len(X);
        n=len(X[0]);
        
#        p=[0 for row in range(m)];
        pred=[model.second for row in range(m)];
        
        if (model.kernelFunction=='linearKernel'):
            p=np.dot(X,model.w)+model.b
#            for i in range(0,m):
#                tmpP=0;
#                for j in range(0,n):
#                    tmpP+=X[i][j]*model.w[j];
#                tmpP+=model.b;
#                p[i]=tmpP;
#                if(tmpP>=0):
#                    pred[i]=model.first;    
        elif(model.kernelFunction=='gaussianKernel'):

            X1=np.sum(X*X, axis=1)
            #print model.X.shape,type(model.X)

            X2=np.sum(model.X*model.X,axis=1)
            #print 'X2',X2.shape
            #print 'X1',X1.shape
            #print 'X',X.shape
            #print 'model.X',model.X.shape
            K=X2.reshape(1,-1)-2*np.dot(X,model.X.T);
            K=X1.reshape(-1,1)+K
            K=np.exp(K*(-model.Sigma))
            K=model.Y.reshape(1,-1)*K
            K=model.alphas.reshape(1,-1)*K
            p=np.sum(K,axis=1)
            
#            for i in range(0,m):
#                prediction=0;
#                for j in range(0,len(model.X)):
#                    tmp=0;
#                    #get ||x-z||**2
#                    for t in range(0,n):
#                        tmp+=(X[i][t] - model.X[j][t])**2;
#                        
#                    tmp=math.exp(-tmp/(2*model.Sigma**2));
#                    prediction=prediction+model.alphas[j]*model.Y[j]*tmp;
#                    
#                prediction+=model.b;
#                p[i]=prediction;
        else:
            print 'no kernel support!' ;
            return -1;
        for i in range(len(p)):
            if p[i]>=0:
                pred[i]=model.first
#        if(prediction>=0):
#                    pred[i]=model.first;
        return (pred,p);
    def predictPricision(self,oriY,preY):
        '''
        it is the total precision
        calculate the precision rate according to the compare of oriY and preY
        '''
        right=0
        for i in range(len(oriY)):
            if oriY[i]==preY[i]:
                right+=1
        print right, len(oriY),len(preY)       
        print "correct rate: ",float(right)/len(preY); 
        return float(right)/len(preY);
    def predictEachClassCriteria(self,oriY,preY):
        '''
        计算每个分类上的分类准确性和召回率,同时也返回总准确率
        problem 怎样处理类别标签为-1的呢？
        '''
        right=0
        oriClass={}
        predictClass={}
        rightClass={} #the labels which is predicted correct.
        for i in range(len(oriY)):
            if oriClass.has_key(oriY[i]):
                oriClass[oriY[i]]=oriClass[oriY[i]]+1
            else:
                oriClass[oriY[i]]=1

            if predictClass.has_key(preY[i]):
                predictClass[preY[i]]=predictClass[preY[i]]+1
            else:
                predictClass[preY[i]]=1

            if oriY[i]==preY[i]:
                right+=1
                if rightClass.has_key(preY[i]):
                    rightClass[preY[i]]=rightClass[preY[i]]+1
                else:
                    rightClass[preY[i]]=1

        #calculate the recall and precison
        result={}
        for label in oriClass.keys():
            if rightClass.has_key(label):
                #cal precision
                if predictClass.has_key(label):
                    precison=float(rightClass[label])/float(predictClass[label]);
                else:
                    precison=-1;
                result[label]=[];
                result[label].append(precison);
                #recall
                recall=float(rightClass[label])/float(oriClass[label])
                result[label].append(recall);
            else:
                result[label]=[0,0];

        for label in result.keys():
            print label,"\t",result[label]
        print "correct rate =",float(right)/len(preY)
        return result;

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
        X=np.array(X)
        
        if len(models)==1:
            predict,pScore=self.predict(models[0],X);
            print 'o yeah'
            return predict
        
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
#                    X=np.array(X)
#                    Y=np.array(Y)
                    model=self.train(X=X,Y=Y,first=first, second=second,Sigma=sigma,C=C,kernelFunction=kernelFunction,tol=tol,max_passes=max_passes)
                    models.append(model)
        print len(models)
        return models
    
    def readDataSeq(self,datasetPath):
        X=[]
        Y=[]
        with open(datasetPath,'r') as f:
            for line in f:
                tmp=line.strip().split('\t')
                Y.append(int(tmp[0].strip()))
                X.append(map(float,tmp[1:]))
        return (X,Y)
    
    def readDataSeqLibsvmForm(self,datasetPath):
        X=[]
        Y=[]
        with open(datasetPath,'r') as f:
            for line in f:
                tmp=line.strip().split(' ')
                if tmp[0].strip()=='':
                    #print "len=",len(tmp);
                    continue
                Y.append(int(tmp[0].strip()))
                tempX=[0.0 for row in range(1000)]
                for element in tmp[1:]:
                    tt=element.strip().split(':')
                    tempX[int(tt[0])-1]=float(tt[1])
               
                X.append(tempX)
        return (X,Y)
       
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
    def DevisionDataAndGridSearch(self,method="linear"):
        '''
        TO-DO 
        you may need to get the trainset and label mapping together, 
        and label should be 1 or -1? or we should modify the train function
        读入测试集和训练集，将一对训练和测试对放入search函数中计算。。。
        把数据分成几个部分，分别输入不同的gridSearch函数中，最后对所得到的数据进行整理
        '''
        trainSetList,testSetList=Read_N_flodData();
        for i in range(0,5):
            trainSet=trainSetList[i]
            testSet=testSetList[i]
            if method=="linear":
                LinearGridSearch(trainSet=trainSet,trainLabel=trainLabel,testSet=testSet,testLabel=testLabel,ParaResultList=ParaResultList,Cstart=-5,Cend=-3,Cstep=2)

    def testSearch(self,method="linear"):
        '''
        firstly get the linear search 效果，看看成效如何
        首先读入train和test两个set，进行测试，最后输出果效
        '''
        #get the data
        (trainSet,trainLabel)=self.readDataSeqLibsvmForm("C:/Users/weiwei/Documents/GitHub/libsvm/tools/train.txt")
        (testSet,testLabel)=self.readDataSeqLibsvmForm("C:/Users/weiwei/Documents/GitHub/libsvm/tools/test.txt")
        ParaResultList={}
        if method=="linear":
            self.LinearGridSearch(trainSet=trainSet,trainLabel=trainLabel,testSet=testSet,testLabel=testLabel,ParaResultList=ParaResultList)
        elif method=="gaussian":
            print "gaussian"
            self.GaussianGridSearch(trainSet=trainSet,trainLabel=trainLabel,testSet=testSet,testLabel=testLabel,ParaResultList=ParaResultList)
        else:
            print "no kernel"
        highAccuracy=0;
        highC=None;
        print "key\ttrainAccuracy\t testAccuracy\t accuracy"
        for key in ParaResultList.keys():
            for paraResult in ParaResultList[key]:
                print str(key)+'\t'+str(paraResult.trainAccuracy)+'\t'+str(paraResult.testAccuracy)+'\t'+str(paraResult.accuracy);
                if paraResult.accuracy>highAccuracy:
                    highAccuracy=paraResult.accuracy
                    highC=key;
        print "the final and best C is "+str(highC)+" accuracy is "+str(highAccuracy);


    def GaussianGridSearch(self,trainSet,trainLabel,testSet,testLabel,ParaResultList,Cstart=9,Cend=10,Cstep=1,SigmaStart=-7,SigmaEnd=-6,SigmaStep=1):
        '''
        思路：吃饭先穿衣，把trainSet和trainLabel输入train函数之中，给他一个超参数组合，
        得到模型，将testSet输入预测函数中得到预测标签，testLabel和预测标签同时输入计算准确度函数
        中得到准确度，同时在训练集上也进行预测计算准确度，计算其平均的准确度。
        最后把这两者都记录下来，作为在这个参数组合下的数据
        传入paraResultList,直接对其进行修改，影响范围在全局所以不需要再传出来了。
        Attention: first and second modify 
        '''
        for C in np.arange(Cstart,Cend,Cstep):
            for Sigma in np.arange(SigmaStart,SigmaEnd,SigmaStep):
                for passes in range(1,20,2):
                    key=str(C)+","+str(Sigma);
                    print key
                    #begin to training
                    model=self.train(X=trainSet,Y=trainLabel,modifyY=True,Sigma=2**Sigma,C=2**C,kernelFunction='gaussianKernel',max_passes=passes)
                    if model.X.size==0:
                        continue;
                    print "first is",model.first,"second is",model.second
                    '''
                    TO-DO
                    '''
                    predictTrain,temp=self.predict(model,trainSet)# i wanna ignore the p score ..
                    predictTest,temp=self.predict(model,testSet)   
                    paraResult=ParaResult()
                    paraResult.trainAccuracy=self.predictPricision(trainLabel, predictTrain)
                    paraResult.testAccuracy=self.predictPricision(testLabel, predictTest)
                    paraResult.accuracy=0.2*paraResult.trainAccuracy+0.8*paraResult.testAccuracy
                    if(ParaResultList.has_key(key)):
                        ParaResultList[key].append(paraResult)
                    else:
                        ParaResultList[key]=[]
                        ParaResultList[key].append(paraResult)


    def LinearGridSearch(self,trainSet,trainLabel,testSet,testLabel,ParaResultList,Cstart=-3.2,Cend=-2.5,Cstep=0.01):                
        '''
        思想同上，不再赘述
        '''
        
        for C in np.arange(Cstart,Cend,Cstep):
            key=C
            #begin to training
            model=self.train(X=trainSet,Y=trainLabel,modifyY=True,C=2**C,max_passes=20)
            print "first is",model.first,"second is",model.second
            '''
            TO-DO
            '''
            predictTrain,temp=self.predict(model,trainSet)# i wanna ignore the p score ..
            predictTest,temp=self.predict(model,testSet)   
            paraResult=ParaResult()
            paraResult.trainAccuracy=self.predictPricision(trainLabel, predictTrain)
            paraResult.testAccuracy=self.predictPricision(testLabel, predictTest)
            paraResult.accuracy=0.2*paraResult.trainAccuracy+0.8*paraResult.testAccuracy
            if(ParaResultList.has_key(key)):
                ParaResultList[key].append(paraResult)
            else:
                ParaResultList[key]=[]
                ParaResultList[key].append(paraResult) 


class ParaResult(object):
    '''
    this class is for store the result of some parameter on some dataset resuslt,
    each result include trainning accuracy and test accuracy;
    accuracy=0.2*trainAccuracy+0.8*testAccuracy;
    最后用dict进行存储得到的数据，多个数据最后再进行平均得到最终的准确度。
    '''
    def __init__(self):
        trainAccuracy=None;
        testAccuracy=None;
        accuracy=None;

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
    #svm.testSearch(method="gaussian")


    '''test the predictEachClassCriteria '''
    (trainSet,trainLabel)=svm.readDataSeqLibsvmForm("C:/Users/weiwei/Documents/GitHub/libsvm/tools/train.txt")
    (testSet,testLabel)=svm.readDataSeqLibsvmForm("C:/Users/weiwei/Documents/GitHub/libsvm/tools/test.txt")
        
    model=svm.train(X=trainSet,Y=trainLabel,modifyY=True,C=2**(-2.77),max_passes=20)
                    
    predictTrain,temp=svm.predict(model,trainSet)# i wanna ignore the p score ..
    predictTest,temp=svm.predict(model,testSet)
    print "train:"
    svm.predictEachClassCriteria(trainLabel,predictTrain)
    print "test:"
    svm.predictEachClassCriteria(testLabel,predictTest)
    '''test end '''

    #dataset=svm.readTrainData("D:\\Project\\Java\\helloWorld\\svmData\\2class exp\\Training docVec.txt")
    #dataset=svm.readTrainData("D:\\Project\\Java\\helloWorld\\svmData\\2class exp\\Training docVec ig.txt")
    #models=svm.multiClassOne2One(dataset=dataset,kernelFunction='gaussianKernel')
    #svm.storeModels(models, 'D:\\Project\\Java\\svm\\model\\2class exp\\gaussian\\ig gaussian models.txt')
    #X=[]
    #Y=[]
    #for key in dataset.keys():
    #    lenY=len(Y)
    #    lenX=len(dataset[key])
    #    X.extend(dataset[key])
    #    Y[lenY:lenY+lenX]=[key for row in range(0,lenX)]
    #preY=svm.multiClassPredict(models, X)
    #svm.predictPricision(Y, preY)
    
    #dataset=svm.readTrainData("D:\\Project\\Java\\helloWorld\\svmData\\2class exp\\Test docVec ig.txt")
    #X=[]
    #Y=[]
    #for key in dataset.keys():
    #    lenY=len(Y)
    #    lenX=len(dataset[key])
    #    X.extend(dataset[key])
    #    Y[lenY:lenY+lenX]=[key for row in range(0,lenX)]
    #preY=svm.multiClassPredict(models, X)
    #svm.predictPricision(Y, preY)

    
#    dataset=svm.readTrainData("D:\\Project\\Java\\helloWorld\\svmData\\smallData\\toy.txt")
#    X,Y=svm.readDataSeq("D:\\Project\\Java\\helloWorld\\svmData\\smallData\\toylinear.txt")
#    svm.train(X, Y, 1, -1, Sigma=0.1, C=1, kernelFunction='gaussianKernel')
##    models=svm.multiClassOne2One(dataset=dataset,max_passes=20)#kernelFunction='gaussianKernel',
#    models=svm.multiClassOne2One(dataset=dataset,kernelFunction='gaussianKernel')
##    svm.storeModels(models, 'D:\\Project\\Java\\svm\\model\\2class exp\\gaussian\\chi gaussian models.txt')
#    models=svm.readModels('D:/Project/Java/svm/model/2class exp/gaussian/chi gaussian models.txt');
###    
    # X=[]
    # Y=[]
    # for key in dataset.keys():
    #     lenY=len(Y)
    #     lenX=len(dataset[key])
    #     X.extend(dataset[key])
    #     Y[lenY:lenY+lenX]=[key for row in range(0,lenX)]
    
    # preY=svm.multiClassPredict(models, X)
    # svm.predictPricision(Y, preY);
##    svm.printPreAndOriCompare(Y, preY, 'D:/Project/Java/svm/result/chi linear precision.txt')
        
