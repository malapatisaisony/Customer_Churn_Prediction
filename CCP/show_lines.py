import sys
p='app.py'
with open(p,'rb') as f:
    lines=f.read().splitlines()
for i in range(460,560):
    if i < len(lines):
        print(i+1, repr(lines[i]))
    else:
        print(i+1,'<no line>')
