#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np

def dct(a):
    result = np.zeros_like(a,dtype='float64')
    xsize=len(a)
    ysize=len(a[0])
    for u in range(xsize):
        for v in range(ysize):
            val = 0.0
            for x in range(xsize):
                for y in range(ysize):
                    val += a[x][y]*np.cos((2*x+1)*u*np.pi/(2*xsize))*np.cos((2*y+1)*v*np.pi/(2*ysize))
            result[u][v] = 4*val
    return result

def idct(a):
    result = np.zeros_like(a,dtype='float64')
    xsize=len(a)
    ysize=len(a[0])
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
                    val += a[u][v] * c1 * c2

            result[x][y] = val
    return result

def main():
    a = np.array([i+1 for i in range(12)])
    a=a.reshape([3,4])
    print(a)
    print(idct(a))


if __name__ == "__main__":
    main()


