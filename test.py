# #定义一个父类
# class faster(object):
#     def __init__(self,name):
#         self.name=name
#         print(' father name',self.name)
#     def Getname(self):
#         return 'father' +self.name
# class son(faster):
#     def __init__(self,name):
#         super(son,self).__init__(name)
#         self.name=name
#         print('son name', self.name)
#     def Getname(self):
#         return 'son name' +self.name
# if __name__=='__main__':
#     son=son('123')
#     print(son.Getname())
a=input().lower()
b=input().lower()
print(a.count(b))
