#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
VERY IMPORTANT: 离散余弦
type=II "The" DCT Method
DON'T FORGET: norm=ortho
'''

class CartoGo():
    def __init__(self, user_rho):
        '''
        这部分的内容：
        注意定义所有的类变量
        傅里叶变化部分：把变量拉长执行fft.dct方法, fft.idct变回原来的矩阵
        注意名字里带fft的变量, 是xsize*ysize
        '''
        self.rho = np.array(user_rho)
        self.xsize = len(user_rho)
        self.ysize = len(user_rho[0])
        self.expky = np.zeros(self.ysize)
        self.fftexpt = np.zeros([self.xsize,self.ysize])
        # self.flat_rho = self.rho.reshape(self.xsize*self.ysize) # rho拉成一阶的
        '''
        WARNING
        '''
        # self.fftrho = fft.dct(user_rho,type=2,norm='ortho') # fftrho: 大小，xsize,ysize
        self.fftrho = self.dct_matrix()
        self.vxt = []
        self.vyt = []
        self.rhot = []
        # 注意：速度向量的范围应该多1
        for _ in range(5):
            this_vxt = np.zeros([self.xsize + 1, self.ysize + 1])
            self.vxt.append(this_vxt)
            this_vyt = np.zeros([self.xsize + 1, self.ysize + 1])
            self.vyt.append(this_vyt)
            this_rhot = np.zeros([self.xsize,self.ysize])
            self.rhot.append(this_rhot)
        self.vxt = np.array(self.vxt)
        self.vyt = np.array(self.vyt)
        self.rhot = np.array(self.rhot)
        self.INITH = 0.001          # Initial size of a time-step
        self.TARGETERROR = 0.01     # Desired accuracy per step in pixels
        self.MAXRATIO = 4.0         # Max ratio to increase step size by
        self.EXPECTEDTIME = 1.0e8   # Guess as to the time it will take
        self.OFFSET = 0.005

        # Inconsistancy check: IGNORED

    def dct_matrix(self):
        xsize=self.xsize
        ysize=self.ysize
        result = np.zeros([xsize,ysize],dtype='float64')
        for u in range(xsize):
            for v in range(ysize):
                val = 0
                for x in range(xsize):
                    for y in range(ysize):
                        val += self.rho[x][y]*np.cos((2*x+1)*u*np.pi/(2*xsize))*np.cos((2*y+1)*v*np.pi/(2*ysize))
                result[u][v] = 4*val
        return result

    def idct_matrix(self):
        xsize=self.xsize
        ysize=self.ysize
        result = np.zeros([xsize,ysize],dtype='float64')
   # 用矩阵表示：首行为1？
        for x in range(xsize):
            for y in range(ysize):
                val = 0.0
                for u in range(xsize):
                    for v in range(ysize):
                        if u==0:
                            c1 = 1
                        else:
                            c1 = 2*np.cos((2*x+1)*u*np.pi/(2*xsize))
                        if v==0:
                            c2 = 1
                        else:
                            c2 = 2*np.cos((2*y+1)*v*np.pi/(2*ysize))
                        val += self.fftexpt[u][v] * c1 * c2
                result[x][y] = val
        return result


    def cart_density(self, t, s):
        for iy in range(self.ysize):
            ky = np.pi*iy/self.ysize;
            self.expky[iy] = np.exp(-ky*ky*t)
        
        # Multiply the FT of the density by the appropriate factors
        for ix in range(self.xsize):
            kx = np.pi*ix/self.xsize
            expkx = np.exp(-kx*kx*t)
            for iy in range(self.ysize):
                self.fftexpt[ix][iy] = expkx*self.expky[iy]*self.fftrho[ix][iy]; # fftexpt: t时刻的密度
               # 公式[5]的傅里叶展开计算形式
               # 展开->逆展开 得到结果，这一步用傅里叶变换
        # fftexpt 逆展开到rhot[s] 
        # self.rhot[s] = fft.idct(self.fftexpt,norm='ortho')
        self.rhot[s] = self.idct_matrix()

    def cart_vgrid(self, s):# 通过rho的场求速度 角用0 边用一维的导数 里面的用二维的 对应公式[6]
        xsize = self.xsize
        ysize = self.ysize
        # 角落值：
        self.vxt[s][0][0] = 0.0
        self.vyt[s][0][0] = 0.0
        self.vxt[s][xsize][0] = 0.0
        self.vyt[s][xsize][0] = 0.0
        self.vxt[s][0][ysize] = 0.0
        self.vyt[s][0][ysize] = 0.0
        self.vxt[s][xsize][ysize] = 0.0
        self.vyt[s][xsize][ysize] = 0.0

        # 上界

        r11 = self.rhot[s][0][0] # REMIND: rhot 表示t时刻的密度
        for ix in range(1,xsize):
            r01 = r11
            r11 = self.rhot[s][ix][0]
            self.vxt[s][ix][0] = -2*(r11-r01)/(r11+r01)
            self.vyt[s][ix][0] = 0.0

        # 下界

        r10 = self.rhot[s][0][ysize-1]
        for ix in range(1,xsize):
            r00 = r10
            r10 = self.rhot[s][ix][ysize-1]
            self.vxt[s][ix][ysize] = -2*(r10-r00)/(r10+r00)
            self.vyt[s][ix][ysize] = 0.0

        # 左边界
        r11 = self.rhot[s][0][0]
        for iy in range(1,ysize):
            r10 = r11
            r11 = self.rhot[s][0][iy]
            self.vxt[s][0][iy] = 0.0
            self.vyt[s][0][iy] = -2*(r11-r10)/(r11+r10)

        # 右边界
        r01 = self.rhot[s][xsize-1][0]
        for iy in range(1,ysize):
            r00 = r01
            r01 = self.rhot[s][xsize-1][iy]
            self.vxt[s][xsize][iy] = 0.0
            self.vyt[s][xsize][iy] = -2*(r01-r00)/(r01+r00)

        # 剩下的所有中间的点
        for ix in range(1,xsize):
            r01 = self.rhot[s][ix-1][0]
            r11 = self.rhot[s][ix][0]
            for iy in range(1,ysize):
                r00 = r01
                r10 = r11
                r01 = self.rhot[s][ix-1][iy]
                r11 = self.rhot[s][ix][iy]
                mid = r10 + r00 + r11 + r01
                self.vxt[s][ix][iy] = -2*(r10-r00+r11-r01)/mid
                self.vyt[s][ix][iy] = -2*(r01-r00+r11-r10)/mid # nabla rho / rho
                # Use Grid to perform gradient

    def cart_velocity(self, rx, ry, s, vxp, vyp):
        '''
        VXP,VYP: We want it to be inplace, thus are lists with only one element
        '''
        xsize = self.xsize
        ysize = self.ysize

        # 继续调节边界条件：
        ix = int(rx)
        if ix<0:
            ix = 0
        else:
            if ix>=xsize:
                ix = xsize - 1

        iy = int(ry)
        if iy<0:
            iy = 0
        else:
            if iy>=ysize:
                iy = ysize - 1
        # Calculate the weights for the bilinear interpolation
        dx = rx - ix
        dy = ry - iy

        dx1m = 1.0 - dx
        dy1m = 1.0 - dy

        w11 = dx1m*dy1m # most case 1
        w21 = dx*dy1m # 0
        w12 = dx1m*dy # 0
        w22 = dx*dy # 0

        # Perform the interpolation for x and y components of velocity

        vxp[0] = w11*self.vxt[s][ix][iy] + w21*self.vxt[s][ix+1][iy] + w12*self.vxt[s][ix][iy+1] + w22*self.vxt[s][ix+1][iy+1]
        vyp[0] = w11*self.vyt[s][ix][iy] + w21*self.vyt[s][ix+1][iy] + w12*self.vyt[s][ix][iy+1] + w22*self.vyt[s][ix+1][iy+1]
    
    def cart_twosteps(self, pointx, pointy, npoints, t, h, s, errorp, drp, spp):
        xsize = self.xsize
        ysize = self.ysize
        s0 = s
        s1 = (s+1)%5
        s2 = (s+2)%5
        s3 = (s+3)%5
        s4 = (s+4)%5
        # print(s2)
        
        # Calculate the density field for the four new time slices
        self.cart_density(t+0.5*h,s1)
        self.cart_density(t+1.0*h,s2)
        self.cart_density(t+1.5*h,s3)
        self.cart_density(t+2.0*h,s4)
        
        # Calculate the resulting velocity grids
        self.cart_vgrid(s1)
        self.cart_vgrid(s2)
        self.cart_vgrid(s3)
        self.cart_vgrid(s4)
        
        # Do all three RK steps for each point in turn
        esqmax = 0.0
        drsqmax = 0.0

        for p in range(npoints):
            rx1 = pointx[p]
            ry1 = pointy[p]
            
            # Do the big combined (2h) RK step
            v1x = [0]
            v1y = [0]
            self.cart_velocity(rx1,ry1,s0,v1x,v1y)
            k1x = 2*h*v1x[0]
            k1y = 2*h*v1y[0]

            v2x = [0]
            v2y = [0]
            self.cart_velocity(rx1+0.5*k1x,ry1+0.5*k1y,s2,v2x,v2y)
            k2x = 2*h*v2x[0]
            k2y = 2*h*v2y[0]
        
            v3x = [0]
            v3y = [0]
            self.cart_velocity(rx1+0.5*k2x,ry1+0.5*k2y,s2,v3x,v3y);
            k3x = 2*h*v3x[0]
            k3y = 2*h*v3y[0]
        
            v4x = [0]
            v4y = [0]
            self.cart_velocity(rx1+k3x,ry1+k3y,s4,v4x,v4y)
            k4x = 2*h*v4x[0]
            k4y = 2*h*v4y[0]

            dx12 = (k1x+k4x+2.0*(k2x+k3x))/6.0
            dy12 = (k1y+k4y+2.0*(k2y+k3y))/6.0

            '''
            Do the first small RK step.  No initial call to cart_velocity() is done
            because it would be the same as the one above, so there's no need
            to do it again
            '''
            
            k1x = h*v1x[0]
            k1y = h*v1y[0]
            
            self.cart_velocity(rx1+0.5*k1x,ry1+0.5*k1y,s1,v2x,v2y)
            k2x = h*v2x[0]
            k2y = h*v2y[0]
            
            self.cart_velocity(rx1+0.5*k2x,ry1+0.5*k2y,s1,v3x,v3y)
            k3x = h*v3x[0]
            k3y = h*v3y[0]
        
            self.cart_velocity(rx1+k3x,ry1+k3y,s2,v4x,v4y)
            k4x = h*v4x[0]
            k4y = h*v4y[0]

            dx1 = (k1x+k4x+2.0*(k2x+k3x))/6.0
            dy1 = (k1y+k4y+2.0*(k2y+k3y))/6.0

            # Do the second small RK step */

            rx2 = rx1 + dx1
            ry2 = ry1 + dy1

            self.cart_velocity(rx2,ry2,s2,v1x,v1y);
            k1x = h*v1x[0]
            k1y = h*v1y[0]
            self.cart_velocity(rx2+0.5*k1x,ry2+0.5*k1y,s3,v2x,v2y)
            k2x = h*v2x[0]
            k2y = h*v2y[0]
            self.cart_velocity(rx2+0.5*k2x,ry2+0.5*k2y,s3,v3x,v3y)
            k3x = h*v3x[0]
            k3y = h*v3y[0]
            self.cart_velocity(rx2+k3x,ry2+k3y,s4,v4x,v4y)
            k4x = h*v4x[0]
            k4y = h*v4y[0]

            dx2 = (k1x+k4x+2.0*(k2x+k3x))/6.0
            dy2 = (k1y+k4y+2.0*(k2y+k3y))/6.0

            # Calculate the (squared) error

            ex = (dx1+dx2-dx12)/15
            ey = (dy1+dy2-dy12)/15
            esq = ex*ex + ey*ey
            if esq>esqmax:
                esqmax = esq

            '''
            Update the position of the vertex using the more accurate (two small
            steps) result, and deal with the boundary conditions.  This code
            does 5th-order "local extrapolation" (which just means taking
            the estimate of the 5th-order term above and adding it to our
            4th-order result get a result accurate to the next highest order)
            '''

            dxtotal = dx1 + dx2 + ex   # Last term is local extrapolation
            dytotal = dy1 + dy2 + ey   # Last term is local extrapolation
            drsq = dxtotal*dxtotal + dytotal*dytotal
            if drsq>drsqmax:
                drsqmax = drsq

            rx3 = rx1 + dxtotal
            ry3 = ry1 + dytotal

            if rx3<0:
                rx3 = 0
            else:
                if rx3>xsize:
                    rx3 = xsize

            if ry3<0:
                ry3 = 0
            else:
                if ry3>ysize:
                    ry3 = ysize

            pointx[p] = rx3
            pointy[p] = ry3

        errorp[0] = np.sqrt(esqmax)
        drp[0] =  np.sqrt(drsqmax)
        spp[0] = s4 # 一个整数


    # Function to estimate the percentage completion
    def cart_complete(self,t):
        res = int(100*np.log(t/self.INITH)/np.log(self.EXPECTEDTIME/self.INITH))
        if res>100:
            res = 100
        return res
    
    def cart_makecart(self, pointx, pointy, npoints, blur):
        xsize = self.xsize
        ysize = self.ysize
        # Calculate the initial density and velocity for snapshot zero 
        self.cart_density(0.0,0)
        self.cart_vgrid(0)
        s = 0
        # Now integrate the points in the polygons
        step = 0
        t = 0.5*blur*blur;
        h = self.INITH;
        error = [0]
        dr = [1]
        sp = [0]

        while dr[0]>0.0 :
            # plt.scatter(pointx,pointy)
            # plt.show()
            # Do a combined (triple) integration step
            self.cart_twosteps(pointx,pointy,npoints,t,h,s,error,dr,sp)
            # Increase the time by 2h and rotate snapshots
            t += 2.0*h
            step += 2
            s = sp[0]

            # Adjust the time-step.  Factor of 2 arises because the target for
            # the two-step process is twice the target for an individual step

            desiredratio = pow(2*self.TARGETERROR/error[0],0.2)

            if desiredratio>self.MAXRATIO:
                h *= self.MAXRATIO
            else:
                h *= desiredratio

            done = self.cart_complete(t)

def main():
    # Insert Code Here...
    rho = [[15 for i in range(10)] for i in range(10)]
    print(rho)
    for i in range(3,6):
        for j in range(3,6):
            rho[i][j] = 25
    # for i in range(20,30):
    # for j in range(10,20):
    # rho[i][j] = 10
    for i in range(6,9):
        for j in range(3,6):
            rho[i][j] = 5
        model = CartoGo(rho)
        i = 0
        x = [0 for i in range(121)]
        y = [0 for i in range(121)]
        for iy in range(11):
            for ix in range(11):
                x[i] = ix
                y[i] = iy
                i+=1
    # for i in range()
    # plt.scatter(x,y)
    # plt.show()
    # print(x,y)
    model.cart_makecart(x,y,121,0)
    print(x)
    print(y)
    plt.scatter(x,y,linewidth=0.5)
    plt.show()



if __name__ == "__main__":
    main()


