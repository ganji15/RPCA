'''
Author: ganji15@mails.ucas.ac.cn
'''

from Res import *
from Utils import *
import numpy
import time

def Inexact_ALM(D, gamma = gamma):
    A = numpy.zeros_like(D)
    E = numpy.zeros_like(D)
    Y = numpy.zeros_like(D)
    #Y = get_init_Y(D, gamma)
    

    p = p0
    u = get_init_u(D)

    for x in xrange(iters):
        #print 'update Ak'
        A = update_Ak(D, E, Y, u)
        #print 'A - D : %.2f'%numpy.linalg.norm(A - D)

        #print 'update Ek'
        Ek = E
        E = update_Ek(D, A, Y, u, gamma)
        #print 'E - D : %.2f'%numpy.linalg.norm(E - D)

        #print 'update Yk'
        Y = update_Yk(Y, D, A, E, u)

        #print 'update uk'
        u = update_uk(p, u, Ek, E, D)

        if is_converged(D, A, E, gamma, x):
            break

    return A, E


def update_uk(p, u, Ek, E, D, e_2 = e_2, max_u = max_u):
    #'''
    err = u * numpy.linalg.norm(E - Ek, ord = 'fro') / numpy.linalg.norm(D, ord = 'fro')
    if err < e_2:
        return min(p * u, max_u * u)
    else:
        return u
    #'''
    #return p * u

def update_Yk(Y, D, A, E, u):
    return Y + u * (D - A - E)

def update_Ak(D, E, Y, u):
    inv_u = 1.0 / u
    M = D - E + inv_u * Y
    U, S, V = numpy.linalg.svd(M, full_matrices=False)
    S = shrink_Diag(S, inv_u)
    S = numpy.diag(S)
    return numpy.dot(U, numpy.dot(S, V))    


def update_Ek(D, A, Y, u, gamma):
    inv_u = 1.0 / u
    M = D - A + inv_u * Y
    return shrink_Mat(M, gamma * inv_u)

def get_init_Y(D, gamma):
    sign_D = (D > 0) * D
    J_sign_D =  max(numpy.linalg.norm(D), 1.0 / gamma * numpy.linalg.norm(D, ord = numpy.inf))
    return sign_D / J_sign_D

def get_init_u(D):
    return 1.0 / numpy.linalg.norm(D, ord = 'fro')

def is_converged(D, A, E, gamma, iter = 0, e_1 = e_1, e_2 = e_2, show_info = True):
    err_Fi = numpy.linalg.norm(D - A - E, ord = numpy.inf)

    if show_info:
        err_F = numpy.linalg.norm(D - A - E, ord = 'fro')        
        err_E = numpy.linalg.norm(E, ord = 1)
        err_J = numpy.linalg.norm(A, ord = 'nuc') + gamma * numpy.linalg.norm(E, ord = 1)
        print 'iter:%3d |D-A-E|f:%8.2f |D-A-E|inf:%8.2f |E|1:%8.2f| err_J:%8.2f'%(iter, err_F, err_Fi, err_E, err_J)
    
    return err_Fi < e_1
    
def shrink_Diag(S, shrink):
    s_num = len(S)
    for i in xrange(s_num):
        if S[i] > shrink:
            S[i] = S[i] - shrink
        elif S[i] < -shrink:
            S[i] = S[i] + shrink
        else:
            S[i] = 0

    return S

def shrink_Mat(W, shrink):
    for i in xrange(W.shape[0]):
        for j in xrange(W.shape[1]):
            if W[i, j] > shrink:
                W[i, j] -= shrink
            elif W[i, j] < -shrink:
                W[i, j] += shrink
            else:
                W[i, j] = 0
    return W

if __name__ == '__main__':
    print 'get imgs files'
    imgs = get_students_imgs(student_number)
    img_num = len(imgs)
    
    print 'load imgs to matrix D'
    D = imgs_to_matrix(imgs)
    
    print 'Inexact ALM begin'
    start = time.time()
    #A, E = Inexact_ALM(D, gamma)
    A, E = Inexact_ALM(D, 0.7)
    end = time.time()
    print 'IALM cost: %2.2fs'%(end - start)
    
    print 'save results'
    numpy.savetxt(APath, A, fmt)
    numpy.savetxt(EPath, E, fmt)