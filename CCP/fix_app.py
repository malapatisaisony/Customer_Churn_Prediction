import io
p = 'app.py'
with open(p, 'rb') as f:
    b = f.read()
s = b.decode('utf-8')
# Fix common broken splits introduced by previous patches
s = s.replace("select\ned_idx", "selected_idx")
s = s.replace("df_loc.sh\nape", "df_loc.shape")
s = s.replace("> 0 e\nlse", "> 0 else")
with open(p, 'w', encoding='utf-8') as f:
    f.write(s)
print('fixed app.py')
