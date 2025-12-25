import ast
p='app.py'
try:
    src=open(p,'r',encoding='utf-8').read()
    ast.parse(src)
    print('AST OK')
except SyntaxError as e:
    print('SyntaxError', e.lineno, e.offset)
    print(repr(e.text))
except Exception as e:
    print('OtherError', e)
