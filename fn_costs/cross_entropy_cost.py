import numpy as np


class CrossEntropyCost:
    @staticmethod
    def fn(y, a):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def fn_d(y, a):
        return a - y


"""
    Cross entropy function derivative with respect to weights:

     df     -y                          1 - y
     --  =  --- * activation_fn'(z) - ------- * - activation_fn'(z) =
     dw      a                          1 - a
    
             -y                         1 - y
         =  --- * activation_fn'(z) + ------- * activation_fn'(z) = 
             a                          1 - a
    
             -y     1 - y                      
         = ( --- + ------- ) * activation_fn'(z) = 
              a     1 - a
    
            -y + ya + a - ya                      
         = ------------------ * activation_fn'(z) = 
              a * (1 - a)
    
                  a - y                      
         =  ------------------- * activation_fn'(z) = 
             activation_fn'(z)
    
         =  a - y
"""
