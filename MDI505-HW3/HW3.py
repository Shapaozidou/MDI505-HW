# -*- coding: utf-8 -*-
"""
@Author  : Shengli Xu
@Contact : shenglix@buffalo.edu
@FileName: HW3.py
@Time    : 2019-02-24 20:24
@Desc    : None
"""

import numpy as np
import matplotlib.pyplot as plt


def energy(a, nc, crystal_type='bcc'):
    if crystal_type == 'fcc':
        n = 4
        r = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    elif crystal_type == 'bcc':
        n = 2
        r = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])

    ucell = 0

    for k in range(-nc, nc):
        for l in range(-nc, nc):
            for m in range(-nc, nc):
                for i in range(n):
                    for j in range(n):
                        dist = a * np.sqrt((k+r[j, 0]-r[i, 0])**2 + (l+r[j, 1]-r[i, 1])**2 + (m+r[j, 2]-r[i, 2])**2)
                        if dist > 0:
                            u = 2 * (1/dist**12 - 1/dist**6)
                        else:
                            u = 0
                        ucell += u

    ucell /= n

    return ucell


def plot(a, u):
    plt.figure(figsize=(10, 10))
    plt.plot(a, u)
    plt.xlabel('a')
    plt.ylabel('u')
    plt.title('energy versus spacing curve')
    plt.show()


a = np.linspace(1, 2, 100)
ucell = []
for space in a:
    ucell.append(energy(space, 6, 'bcc'))

plot(a, ucell)


