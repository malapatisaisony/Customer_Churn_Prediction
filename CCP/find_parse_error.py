import ast
p='app.py'
s=open(p,'r',encoding='utf-8').read()
lines=s.splitlines()
for i in range(1,len(lines)+1):
    try:
        ast.parse('\n'.join(lines[:i]))
    except SyntaxError as e:
        print('failed at', i, '->', repr(lines[i-1]))
        print('msg', e)
        break
else:
    print('parsed full file OK')
