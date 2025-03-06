import re

list_of_words = []

filename = "text_file.txt"

with open(filename, "r") as f:
    for line in f:
        list_of_words.extend(re.findall(r"[\w]+", line.lower()))

        
unique = {}
for each in list_of_words:
    if each not in unique:
        unique[each] = 0
    unique[each] += 1
 
s = []

for key, val in unique.items():
    if val == 1:
        s.append(key)
        
print(sorted(s))
    

