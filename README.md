# CartoGo
## 样图展示
![Sample Image of US Cartogram](https://github.com/Cleophile/CartoGo/blob/master/carto_large_result.jpg)  

## CartoGo使用方法：
1. 根据获取到的数据，如GDP、人口、星巴克门店个数等等，放入到`us_density.csv`表格中的`total`列下。  
2. 运行`physics_construct.py`，获得变换后的点，这个过程需要耗时几十分钟-1小时，取决于密度的均匀程度  
3. 运行`draw.py`, 获得最终的图形，这个过程需要耗时2小时  
**注意**：Cartogram生成是一个复杂的算法，就目前找到的论文来看，我们的图像精度更高，稀疏采样几十分钟的物理建模属于较快的算法  

## 文件说明：
### 代码部分
`drill_carto`: 存放CartoGo类，用于计算不同密度点的扩散  
`ThinPlate`: 存放ThinPlate类，用于根据扩散好的点集恢复图像  
`physics_construct`: 用于读取文件并运行`drill_carto`  
`draw`: 用于读取文件并生成图形  

### 数据文件
`us_density`: 用户用于存放密度的文件，内置的数据为美国各州GDP，可更改  
`us_large.jpg`: 一个较清晰的美国地图  
`carto_large_result`: 已经运行好的一个范例图片  
`75_30_result`: 已经运行好的一个范例物理建模后的点集  

### Sampling文件夹
1. 存放了采样、作图的算法和原始文件  
2. 存放了采样的结果  

### pre_keynote文件夹
包括了presentation的PDF版本的keynote  

### 参考文献文件夹
包括了所有的参考文献，具体可参考report

## 期望改进
1. python二维DCT实现  
2. 

## 联系方式
如果有建议或疑问，请联系
Wang Tianmin [1607130090@fudan.edu.cn](mailto:16307130090@fudan.edu.cn)


