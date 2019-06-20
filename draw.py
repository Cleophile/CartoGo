# IPython log file
import numpy as np
import pandas as pd
import ThinPlate
from PIL import Image

df = pd.read_csv('75_30_result.csv')
df['dx'] = df['result_x'] - df['ori_x']
df['dy'] = df['result_y'] - df['ori_y']
df['motion'] = df['dx']**2 + df['dy']**2
df=df.sort_values('motion',ascending=False)
    # for i in df.index[:350]:
# plt.scatter(df.loc[i,'result_x'],df.loc[i,'result_y'])

ori_point = []
result_point = []
for i in df.index:
    ori_point.append((df.loc[i,'ori_x'],df.loc[i,'ori_y']))
    result_point.append((df.loc[i,'result_x'],df.loc[i,'result_y']))

print('Point sorted!')

model=ThinPlate.ThinPlate(before_cors=result_point,after_cors=ori_point)
model.regress()

print('Model Built')

us_geo = Image.open('us_large.jpg')
us_geo_pix = us_geo.load()
(xsize,ysize) = us_geo.size

new_img = np.zeros([ysize,xsize,3])

for i in range(xsize):
    print('Doing rotation:',i)
    for j in range(ysize):
        u = i/xsize*75
        v = (ysize-1-j)/ysize*30
        target_point = model.predict(u,v)
        x = int(target_point[0]*xsize/75)
        y = ysize-1-int(target_point[1]*ysize/30)
        if y>=0 and y<ysize and x>=0 and x<xsize:
            new_img[j][i][0] = us_geo_pix[x,y][0]
            new_img[j][i][1] = us_geo_pix[x,y][1]
            new_img[j][i][2] = us_geo_pix[x,y][2]

result_img = Image.fromarray(new_img.astype('uint8'))
result_img.save('carto_large_result.jpg')
# result_img.show()
