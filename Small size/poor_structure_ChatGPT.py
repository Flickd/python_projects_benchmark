numbs = [1,2,3,4,5,6,7,8,9,10]
total = 0
index = 0

while index < len(numbs):
    total = total + numbs[index]
    index=index+1

print("the sum is:", total)

total2 = 0
for i in range(len(numbs)):
    total2 = total2 + numbs[i]
print("Sum again:", total2)

def stuff(a):
    result = 0
    for x in a:
        result = result + x
    return result

print("Sum one more time:", stuff(numbs))
