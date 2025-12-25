p='app.py'
lines=open(p,'r',encoding='utf-8').read().splitlines()
stack=[]
for i,line in enumerate(lines, start=1):
    stripped=line.lstrip('\t ')
    indent=len(line)-len(stripped)
    if stripped.startswith('try:'):
        stack.append((i,indent,line))
    elif stripped.startswith('except'):
        # find nearest stack entry with same indent
        for j in range(len(stack)-1, -1, -1):
            if stack[j][1]==indent:
                stack.pop(j)
                break
        else:
            print('Unmatched except at', i)

if stack:
    print('Unmatched try(s):')
    for t in stack:
        print(t)
else:
    print('All tries matched')
