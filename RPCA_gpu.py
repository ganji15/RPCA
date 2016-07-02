from Res import *
from Utils import *
import theano
import theano.tensor as T
import time

def Inexact_ALM_gpu(D, gamma):
    A = theano.shared(numpy.zeros_like(D, dtype = theano.config.floatX), name = 'A')
    E = theano.shared(numpy.zeros_like(D, dtype = theano.config.floatX), name = 'E')
    pre_E = theano.shared(numpy.zeros_like(D, dtype = theano.config.floatX), name = 'pre_E')
    Y = theano.shared(get_init_Y(D, gamma), name = 'Y')

    p = p0
    u = theano.shared(get_init_u(D))
    D = theano.shared(D, name = 'D')

    updates = [(A, update_Ak(D, E, Y, u)), 
                (pre_E, E),
                (E, update_Ek(D, A, Y, u, gamma)),
                (Y, update_Yk(Y, D, A, E, u)),
                (u, update_uk(p, u, pre_E, E, D))]

    cost = T.nlinalg.norm(D - A - E, ord = 'inf')
    update_A = theano.function(inputs = [], outputs = [], updates = [updates[0]])
    update_pre_E = theano.function(inputs = [], outputs = [], updates = [updates[1]])
    update_E = theano.function(inputs = [], outputs = [], updates = [updates[2]])
    update_Y = theano.function(inputs = [], outputs = [], updates = [updates[3]])
    update_u = theano.function(inputs = [], outputs = cost, updates = [updates[4]])
    for x in xrange(iters):
        update_A()
        update_pre_E()
        update_E()
        update_Y()
        cost = update_u() 
        if cost < e_1:
             break

        print 'iter %2d: cost:%.5f'%(x, cost)

    return A, E

def update_uk(p, u, pre_E, E, D, e_2 = e_2, max_u = max_u):
    err = u * T.nlinalg.norm(E - pre_E, ord = 'fro') / T.nlinalg.norm(D, ord = 'fro')
    if err.eval() < e_2:
        return min(p, max_u) * u
    else:
        return u

def update_Yk(Y, D, A, E, u):
    return T.cast(Y + u * (D - A - E), theano.config.floatX)  

def update_Ak(D, E, Y, u):
    inv_u = 1.0 / u
    M = D - E + inv_u * Y
    U, S, V = T.nlinalg.svd(M, full_matrices=False)
    S = shrink(S, inv_u)
    S = T.diag(S)
    return T.cast(T.dot(U, T.dot(S, V)), theano.config.floatX)    


def update_Ek(D, A, Y, u, gamma):
    inv_u = 1.0 / u
    M = D - A + inv_u * Y
    return  T.cast(shrink(M, gamma * inv_u), theano.config.floatX)    

def get_init_u(D):
    return 1.0 / numpy.linalg.norm(D, ord = 'fro')

def get_init_Y(D, gamma):
    sign_D = (D > 0) * D
    J_sign_D =  max(numpy.linalg.norm(D), 1.0 / gamma * numpy.linalg.norm(D, ord = numpy.inf))
    return sign_D / J_sign_D
    
def shrink(M, shrink):
    return T.nnet.relu(M - shrink) - T.nnet.relu(-M - shrink)

if __name__ == '__main__':
    print 'get imgs files'
    imgs = get_students_imgs(student_number)
    img_num = len(imgs)
    
    print 'load imgs to matrix D'
    D = imgs_to_matrix(imgs)
    
    print 'Inexact ALM begin'
    start = time.time()
    A, E = Inexact_ALM_gpu(D, gamma)
    end = time.time()
    print 'IALM cost: %2.2fs'%(end - start)
    
    print 'save results'
    numpy.savetxt(APath, A.eval(), fmt)
    numpy.savetxt(EPath, E.eval(), fmt)