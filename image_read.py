# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:05:06 2024

@author: coby

grids references

https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
### read image file into N x N array
### if image has color then it is N x N x 3

plt.rcParams['figure.figsize'] = [10, 10]
#img = mpimg.imread('figures/50/50.jpg')
img_50 = mpimg.imread('figures/50/50.jpg')
img_75 = mpimg.imread('figures/75/75.jpg')
img_125 = mpimg.imread('figures/125/125.jpg')
img_150 = mpimg.imread('figures/150/150.jpg')
img_200 = mpimg.imread('figures/200/200.jpg')

def slice_img(img, X1, X2, Y, ticks, title):
    plt.figure()
    imgplot = plt.imshow(img)
    img_y, img_x = np.shape(img)
    #print(img_x)
    
    major_ticks = np.arange(int(img_x/2 - img_x*X1), int(img_x/2 + img_x*X2), ticks)

    plt.grid()
    plt.xlabel("Array Index (x)")
    plt.ylabel("Array Index (y)")
    plt.title("{} Lens Image".format(title))

    fig, ax = plt.subplots(2, 1, figsize = (10, 8))
    ax[1].set_xlabel("Array Index (x)")
    ax[0].set_ylabel("Intensity Value \n from Camera")
    ax[1].set_ylabel("Intensity Value Subset \n from Camera")

    ax[0].scatter(np.arange(0, img_x, 1), img[Y], s = 3)
    ax[1].scatter(np.arange(0, img_x, 1)[int(img_x/2 - img_x*X1):int(img_x/2 + img_x*X2)], img[Y][int(img_x/2 - img_x*X1):int(img_x/2 + img_x*X2)], s = 2)

    ax[1].set_xticks(major_ticks)
    ax[1].grid()

    ax[0].set_title("Intensity vs. Array Index \n for a horizontal slice of the image \n using f = {} lens".format(title))

slice_img(img_50, 1/12, 1/16, 1750, 20, "50mm")
slice_img(img_75, 1/20, 1/12, 1500, 20, "75mm")
slice_img(img_125, 1/10, 1/20, 1750, 20, "125mm")
slice_img(img_150, 1/12, 1/20, 1600, 20, "150mm")
slice_img(img_200, 1/18, 1/14, 1600, 20, "200mm")



#slice_img(img_75, 200)



# ### plot image
# imgplot = plt.imshow(img)
# plt.show()a

# #np.shape(img)
# img_x = np.shape(img)[1]
# img_y = np.shape(img)[0]

# zoom = 300

# plt.imshow(img[zoom+int(img_y/4):int(3*img_y/4)-zoom, int(zoom+img_x/4):int(3*img_x/4)-zoom])
# plt.grid(True)

# plt.figure(figsize=(10, 5))
# plt.scatter(np.arange(0, img_x, 1), img[1000], s = 3)

# plt.figure(figsize=(10, 5))
# plt.scatter(np.arange(0, img_x, 1)[500:800], img[1000][500:800], s = 2)

# major_ticks = np.arange(500, 801, 20)

# plt.grid()
# plt.xlabel("Array Index")
# plt.ylabel("Assigned Intensity by the Camera")
# plt.title("Intensity vs. Array Index \n for a horizontal slice of the image")

# fig, ax = plt.subplots(2, 1, figsize = (10, 8))
# ax[1].set_xlabel("Array Index")
# ax[1].set_ylabel("Intensity Value \n from Camera")
# ax[0].set_ylabel("Intensity Value Subset \n from Camera")

# ax[0].scatter(np.arange(0, img_x, 1), img[1000], s = 3)
# ax[1].scatter(np.arange(0, img_x, 1)[500:800], img[1000][500:800], s = 2)

# ax[1].set_xticks(major_ticks)
# ax[1].grid()

# ax[0].set_title("Intensity vs. Array Index \n for a horizontal slice of the image")



