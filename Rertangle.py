# _*_ coding: utf-8 _*_

import numpy as np


def crops(img, img1, crop_shape=[48,48,48]):

    x1, x2, x3 = img.shape
    x_f = []
    x_b = []
    for j in range(x2):
        if np.all(img1[:, j, :] == 0):
            x_f.append(j)
        else:
            x_b.append(j)

    index1 = x_b[0]
    index2 = x_b[-1]
    y = (index1+index2)//2
    y_zuo = max(0, y - (crop_shape[1]//2))
    y_you = y_zuo + crop_shape[1] - 1
    if y_you >= x2:
        y_you = x2-1
        y_zuo = x2-crop_shape[1]
    y_del = list(range(y_zuo))+list(range(y_you+1,x2))
    y1 = np.delete(img, y_del, axis=1)
    y2 = np.delete(img1, y_del, axis=1)

    y_f = []
    y_b = []
    for k in range(x3):
        if np.all(y2[:, :, k] == 0):
            y_f.append(k)
        else:
            y_b.append(k)
    index3 = y_b[0]
    index4 = y_b[-1]
    z = (index3 + index4) // 2
    z_zuo = max(0, z - (crop_shape[2] // 2))
    z_you = z_zuo + crop_shape[2] - 1
    if z_you >= x3:
        z_you = x3-1
        z_zuo = x3-crop_shape[2]
    z_del = list(range(z_zuo)) + list(range(z_you + 1, x3))
    z1 = np.delete(y1, z_del, axis=2)
    z2 = np.delete(y2, z_del, axis=2)

    layer_z = []
    for m in range(x1):
        if z2[m, :, :].sum() > 0:   # 没有勾画的地方sum为0 ，没勾画的地方即为没有肿瘤
            layer_z.append(m)     # layer中存放的为勾画的层
    index5 = layer_z[0]
    index6 = layer_z[-1]
    x = (index5 + index6) // 2
    x_zuo = max(0, x - (crop_shape[0] // 2))
    x_you = x_zuo + crop_shape[0] - 1
    if x_you >= x1:
        x_you = x1-1
        x_zuo = x1-crop_shape[0]
    x_del = list(range(x_zuo)) + list(range(x_you + 1, x1))

    index_z = []
    for k in range(x1):
        if k not in layer_z:
            index_z.append(k)
    img_ori = np.delete(z1, x_del, axis=0)
    mask_ori = np.delete(z2, x_del, axis=0)
    return img_ori, mask_ori, x_zuo, x_you, y_zuo, y_you, z_zuo, z_you

# def crops(img, mask):
#     x1, x2, x3 = img.shape
#
#     layersori_x = []
#     for l in range(x2):
#         if mask[:, l, :].sum() > 0:   # 没有勾画的地方sum为0 ，没勾画的地方即为没有肿瘤
#             layersori_x.append(l)     # layer中存放的为勾画的层
#     index1 = layersori_x[0]
#     index2 = layersori_x[-1]
#
#     indexx = []
#     for l in range(x2):
#         if l not in layersori_x:
#             indexx.append(l)
#     img_x = np.delete(img, indexx, axis=1)
#     mask_x = np.delete(mask, indexx, axis=1)
#
#     layersori_y = []
#     for l in range(x3):
#         if mask_x[:, :, l].sum() > 0:   # 没有勾画的地方sum为0 ，没勾画的地方即为没有肿瘤
#             layersori_y.append(l)     # layer中存放的为勾画的层
#     index3 = layersori_y[0]
#     index4 = layersori_y[-1]
#
#     indexy = []
#     for l in range(x3):
#         if l not in layersori_x:
#             indexy.append(l)
#     img_y = np.delete(img_x, indexy, axis=2)
#     mask_y = np.delete(mask_x, indexy, axis=2)
#
#     layersori_z = []
#     for l in range(x1):
#         if mask_y[l, :, :].sum() == 0:   # 没有勾画的地方sum为0 ，没勾画的地方即为没有肿瘤
#             layersori_z.append(l)     # layer中存放的为勾画的层
#
#     index5 = layersori_z[0]
#     index6 = layersori_z[-1]
#     indexz = []
#     for l in range(x1):
#         if l not in layersori_z:
#             indexz.append(l)
#     img_ori = np.delete(img_y, indexz, axis=0)
#     mask_ori = np.delete(mask_y, indexz, axis=0)
#     return img_ori, mask_ori, index1, index2, index3, index4, index5, index6
