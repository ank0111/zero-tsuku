import numpy as np
from dataset.mnist import load_mnist
from util import im2col_myown, col2im_myown
from sample.common.util import im2col,col2im

(_, _), (t, _) = load_mnist(flatten=False)

t=t[:10,:,10:13,10:13]
t2=np.insert(t[:3],1,t[6],axis=1)
t3 = im2col_myown(t2, 2, 2,1,3)
print(t2.shape)
print(t3.shape)
t4=col2im_myown(t3,t2.shape,2,2,1,3)
t5=col2im(t3,t2.shape,2,2,1,3)
print(np.all(t4==t5))