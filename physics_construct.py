# IPython log file
import numpy as np
import pandas as pd
import json
from drill_carto import CartoGo

xsize = 75
ysize = 30

rho = np.zeros([xsize,ysize])
with open('./sampling/samples_75x30.json','r') as f:
    content = f.read()

sample = json.loads(content)
sample = sample['samples']

t = 0
for i in range(xsize):
    for j in range(ysize):
        rho[i][j] = sample[t][2]
        t+=1

df = pd.read_csv('us_density.csv')
df.loc[49,'area'] = sum(df['area'][:49])
df.loc[49,'total'] = sum(df['total'][:49])
df['density'] = df['total']/df['area']
print(df)

t = 0
for i in range(xsize):
    for j in range(ysize):
        code = rho[i][j]
        if code==99:
            code=49
        rho[i][j] = df.loc[code,'density']


model = CartoGo(rho)

i=0
tsize = (xsize+1)*(ysize+1)
x = [0 for i in range(tsize)]
y = [0 for i in range(tsize)]
for iy in range(ysize+1):
    for ix in range(xsize+1):
        x[i] = ix
        y[i] = iy
        i+=1

model.cart_makecart(x,y,tsize)

# plt.scatter(x,y,linewidths=0.02)
result_x = x.copy()
result_y = y.copy()
i=0
x = [0 for i in range(tsize)]
y = [0 for i in range(tsize)]
for iy in range(ysize+1):
    for ix in range(xsize+1):
        x[i] = ix
        y[i] = iy
        i+=1

df = pd.DataFrame(dict(ori_x=x,ori_y=y,result_x=result_x,result_y=result_y))
df.to_csv('75_30_result.csv')

'''
计算motion画出直方图
motion = []
for i in range(tsize):
    this_move_x = x[i] - result_x[i]
    this_move_y = y[i] - result_y[i]
    this_move = this_move_x*this_move_x + this_move_y*this_move*y
    motion.append(this_move)
    
for i in range(tsize):
    this_move_x = x[i] - result_x[i]
    this_move_y = y[i] - result_y[i]
    this_move = this_move_x*this_move_x + this_move_y*this_move_y
    motion.append(this_move)
    
    
len(motion)
max(motion)
min(motion)
plt.hist(motion)
df.head()
df['dx'] = df['result_x'] - df['ori_x']
df['dy'] = df['result_y'] - df['ori_y']
df['motion'] = df['dx']*df['dx'] + df['dy']*df['dy']
df
df.sort_values('motion')
get_ipython().run_line_magic('ls', '')
df
sdf=df.sort_values('motion',ascending=False)
sdf.head()
sdf.loc[sdf.index[100],motion]
sdf.head()
sdf.loc[sdf.index[100],'motion']
sdf.loc[sdf.index[350],'motion']
sdf.loc[sdf.index[500],'motion']
sdf.loc[sdf.index[600],'motion']
'''

