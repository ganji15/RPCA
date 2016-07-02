'''
Author: ganji15@mails.ucas.ac.cn
'''

from Res import *
from Utils import *
import numpy

E = numpy.loadtxt(EPath)
A = numpy.loadtxt(APath)

imgs = get_students_imgs(student_number)
save_path = os.path.abspath(SrcResultPath)

for i in range(0, len(imgs)):
	
    im = Image.open(imgs[i])
    im.save(save_path + '\\%02dD.jpg'%i)
    im = arr_to_img(A[:, i], [192, 168])
    im.convert('RGB').save(save_path + '\\%02dA.jpg'%i)
    #im.show()
    im = arr_to_img(E[:, i], [192, 168])
    #im.show()
    im.convert('RGB').save(save_path + '\\%02dE.jpg'%i)