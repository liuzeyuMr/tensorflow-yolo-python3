n=list(input())
# n.reverse()
a=[]
for i in n:
    if i in a:
        pass
    else:
        a.append(i)
print(len(a))