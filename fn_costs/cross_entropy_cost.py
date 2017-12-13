import numpy as np


class CrossEntropyCost:
    @staticmethod
    def fn(y, a):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def fn_d(y, a, z, a_fn):
        return a - y


"""
    Cross entropy function derivative with respect to weights:

     df     -y                1 - y
     --  =  --- * a_fn'(z) - ------- * - a_fn'(z) =
     dw      a                1 - a
    
            -y                1 - y
         =  --- * a_fn'(z) + ------- * a_fn'(z) = 
             a                1 - a
    
             -y     1 - y                      
         = ( --- + ------- ) * a_fn'(z) = 
              a     1 - a
    
            -y + ya + a - ya                      
         = ------------------ * a_fn'(z) = 
              a * (1 - a)
    
              a - y                      
         =  ---------- * a_fn'(z) = 
             a_fn'(z)
    
         =  a - y
"""
