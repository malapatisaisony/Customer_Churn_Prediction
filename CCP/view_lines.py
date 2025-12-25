fpath = 'app.py'
with open(fpath, 'rb') as f:
    lines = f.read().splitlines()
for i in range(250,275):
    if i < len(lines):
        print(i+1, repr(lines[i]))
    else:
        print(i+1, '<no line>')
