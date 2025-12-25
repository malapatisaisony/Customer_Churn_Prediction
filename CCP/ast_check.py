import ast
s=open('app.py',encoding='utf-8').read()
try:
    ast.parse(s)
    print('AST ok')
except Exception as e:
    import traceback
    traceback.print_exc()
