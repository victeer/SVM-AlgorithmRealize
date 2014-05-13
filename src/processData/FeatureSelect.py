'''
Created on 2014-3-25

@author: Victor
'''

class MyClass(object):
    '''
    classdocs
    '''
    

    def __init__(self,params):
        '''
        Constructor
        '''
        
if __name__=="__main__":
    from jpype import *
    import os.path
    str=r"D:\Program Files\Java\jdk1.6.0_39\jre\bin\server\jvm.dll"
    startJVM(str, "-ea")
    java.lang.System.out.println("hello World")
    shutdownJVM()
        