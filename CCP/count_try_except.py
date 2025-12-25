p='app.py'
s=open(p,'r',encoding='utf-8').read()
print('tries', s.count('try:'))
print('excepts', s.count('except'))
for i, line in enumerate(s.splitlines(), start=1):
	if 'try:' in line:
		print('try at', i, repr(line))
	if 'except' in line:
		print('except at', i, repr(line))
