'''
Author: ganji15@mails.ucas.ac.cn
'''

import os
import Image
import numpy
from Res import ImageSrcDir, DPath, fmt

def get_student_dir(number, ImageSrcDir = ImageSrcDir):
    return ImageSrcDir + 'yaleB%02d\\'%number


def get_files_of_dir(dir, file_type = '.pgm', is_full_path = True):
    res = []

    for root, dirs, files in os.walk(dir):
        for f in files:
            if not f.endswith(file_type):
                continue
            if is_full_path:
                res.append(dir + f)
            else:
                res.append(f)

    return res


def get_students_imgs(number):
    return get_files_of_dir( get_student_dir(number))


def imgs_to_matrix(imgs):
    img_num = len(imgs)
    im = Image.open(imgs[0]).convert('L')
    im_arr = numpy.array(im)
    rows = im_arr.size
    
    X = numpy.zeros([rows, img_num], dtype = 'float32')
    for i in xrange(img_num):
        X[:, i] = numpy.array( Image.open(imgs[i]).convert('L')).flatten()
    
    return X

def arr_to_img(arr, shape):
    arr = arr.reshape(shape)
    im = Image.fromarray(arr)
    return im

def save_D(student_number):
    print 'get imgs files'
    imgs = get_students_imgs(student_number)
    img_num = len(imgs)
    
    print 'load imgs to matrix D'
    D = imgs_to_matrix(imgs)
    
    print 'save number%2d D.txt'%student_number
    numpy.savetxt(DPath, D, fmt = fmt)