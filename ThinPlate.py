#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise
import numpy as np
from PIL import Image
# import pandas as pd

def u_func(x):
    pass

'''
class DimensionError(Exception):
    def __init__(self):
        self.super().__init__()
'''

class ThinPlate(object):
    def __init__(self,before_cors,after_cors):
        self.before_cors = before_cors
        self.after_cors = after_cors
        self.dims = len(before_cors)
        assert len(after_cors)==self.dims
        self.model_x = None
        self.model_y = None
    
    def regress(self):
        '''
        Model 1: model for x
        Model 2: model for y
        注意共线性的问题
        目前无法解决共线性的问题
        '''
        self.L = np.array([[0.0 for _ in range(self.dims+3)] for _ in range(self.dims+3)])
        for i in range(self.dims):
            for j in range(i+1,self.dims):
                r2 = np.square(self.before_cors[i][0] - self.before_cors[j][0]) + np.square(self.before_cors[i][1] - self.before_cors[j][1])
                val = r2*np.log(r2)
                self.L[i][j] = val
                self.L[j][i] = val

            self.L[self.dims][i] = 1
            self.L[i][self.dims] = 1

            self.L[self.dims+1][i] = self.before_cors[i][0]
            self.L[i][self.dims+1] = self.before_cors[i][0]
                
            self.L[self.dims+2][i] = self.before_cors[i][1]
            self.L[i][self.dims+2] = self.before_cors[i][1]

        # print(self.L)
        # self.Yx = [x1',x2',...,xn',0,0,0].T
        # self.Yy = [y1',y2',...,yn',0,0,0].T
        self.Yx = [i[0] for i in self.after_cors]
        self.Yx.extend([0,0,0])
        self.Yy = [i[1] for i in self.after_cors]
        self.Yy.extend([0,0,0])
        self.Yx = np.array(self.Yx).T
        self.Yy = np.array(self.Yy).T
        # print(self.Yx,self.Yy)
        self.model_x = np.dot(np.linalg.inv(self.L),self.Yx)
        # print(self.model_x)
        self.model_y = np.dot(np.linalg.inv(self.L),self.Yy)
        # print(self.model_y)

    def predict(self,x,y):
        if self.model_x is None:
            raise TypeError('Model hasn\'t been regressed!')
        point_vec = [np.square(x0-x)+np.square(y0-y) for (x0,y0) in self.before_cors]
        point_vec = [f*np.log(f + 1e-6) for f in point_vec]
        point_vec.extend([1,x,y])
        point_vec = np.array(point_vec)
        # print(point_vec)
        # print([np.dot(point_vec,self.model_x), np.dot(point_vec, self.model_y)])
        return [np.dot(point_vec,self.model_x), np.dot(point_vec, self.model_y)]

    def reverse(self,x,y):
        # \sum wiU((x0,y0)-(xi,yi)) + a1 + ax*x0 + ay*y0 = x
        pass

def Parse(ApeImagePath,ApePoints,HumanImagePath,HumanPoints):
    # Model Regress: Ape to Human
    # Unknown Image one pix -> predict to the HumanImage
    ori_human = Image.open(HumanImagePath).convert('L') # 灰度值
    ori_human_pix = ori_human.load()
    human_size = ori_human.size
    # print(np.array(ori_human_pix))
    # print(human_size)

    ori_ape = Image.open(ApeImagePath).convert('L') # 灰度值
    ori_ape_pix = ori_ape.load()
    ape_size = ori_ape.size
    
    model = ThinPlate(before_cors=ApePoints,after_cors=HumanPoints)
    model.regress()
    # New Image is of Ape Size, Ape to human

    new_img = np.array([[0 for _ in range(ape_size[0])] for _ in range(ape_size[1])],dtype='uint8')
    # print(ape_size)
    for i in range(ape_size[0]):
        for j in range(ape_size[1]):
            this_human_point = np.round(model.predict(i,j))
            # color originated from the original human point
            if this_human_point[0]>=0 and this_human_point[0]<human_size[0] and this_human_point[1]>=0 and this_human_point[1]<human_size[1]:
                val = ori_human_pix[this_human_point[0],this_human_point[1]]
                # print(val)
                # print(j,i)
                new_img[j][i] = val
    result_img = Image.fromarray(new_img)
    result_img.show()
    return result_img


if __name__ == "__main__":
    # obj = ThinPlate(before_cors=[[3,3],[6,7]],after_cors=[[4,4],[5,5]])
    # obj.regress()
    # obj.predict(3.5,3.5)
    ApeImagePath = '/Users/wangtianmin/Downloads/ape.png'
    ApePoints = [[68,25],
            [109,37],
            [152,32],
            [186,29],
            [127,33],
            [80,198],
            [165,204],
            [61,172],
            [77,211],
            [125,231],
            [180,212],
            [198,189]]

    HumanImagePath = '/Users/wangtianmin/Downloads/王天民_Tim_一寸.jpg'
    HumanPoints = [[295,438],
            [368,439],
            [445,438],
            [515,435],
            [404,424],
            [359,536],
            [450,537],
            [341,596],
            [377,606],
            [406,611],
            [442,604],
            [477,604]]

    Parse(ApeImagePath,ApePoints,HumanImagePath,HumanPoints)

    


