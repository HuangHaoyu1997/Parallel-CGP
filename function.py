import numpy as np
import math
import operator as op
import sympy as sp

class Function:
    """
    A general function
    arity: 函数的输入参数的数量
    """

    def __init__(self, f, arity, name=None):
        self.f = f
        self.arity = arity
        self.name = f.__name__ if name is None else name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def protected_div(a, b, epsilon=1e-3):
    if abs(b) < epsilon:
        return a / (b+epsilon)
    else:
        return a / b

def sqrt(a):
    '''sqrt(abs(a))'''
    return math.sqrt(abs(a))

def relu(a):
    if isinstance(a,complex): print(a)
    if a>=0: return a
    else: return 0

def ln(a, epsilon=1e-3):
    if abs(a) <= epsilon: return 0
    else: return math.log(abs(a))

def exp(a):
    a = np.clip(a, -100, 100)
    return (np.exp(a)-1)/(np.exp(1)-1)

def max1(a):
    return max(a,0) # ,sp.maximum(a,sp.sqrt(1))

def min1(a):
    '''min1是冗余的,可以用op.neg+max1来实现'''
    return min(a,0)

def max2(a,b):
    if a <= b: return b
    else: return a

def min2(a,b):
    if a <= b: return a
    else: return b

def scaled(a):
    '''a压缩到[-1,1]区间'''
    if a is None: return 0.0
    return min(max(a, -1.0), 1.0)

def sign(a):
    '''其实是1-sign(x)'''
    # 注意！！！！！！
    # 8月15日之前采用的是与下面相反的逻辑
    if a>0: return 1
    else: return 0

def inv(a, epsilon=1e-3):
    if abs(a) <= epsilon: return 1
    else: return 1/a

def beta(a,b): return np.random.beta(a,b) # Beta分布,[0,1]区间
def uniform(a,b): return np.random.uniform(a,b) # [a,b]均匀分布
def uniform01(): return np.random.uniform(0,1) # [0,1]均匀分布

def Add(x,y): return np.add(x,y,dtype=np.float)
def Sub(x,y): return np.subtract(x,y,dtype=np.float)
def Mul(x,y): return np.maximum(x,y) # 向量按位做乘法. 标量-标量,向量-向量,向量-标量,标量-向量,均合法
def Div(x,y): return np.divide(x,y) # 向量按位做除法. 标量-标量,向量-向量,向量-标量,标量-向量,均合法
def Inv(x): return np.reciprocal(x, dtype=np.float) # 必须是浮点数
def Neg(x): return -x
def Sum(x): return np.sum(x)
def MAX(x): return np.max(x)
def Max1(x): return np.maximum(x,0.)
def Min1(x): return np.minimum(x,0.)
def Max2(x,y): return np.maximum(x,y)
def Min2(x,y): return np.minimum(x,y)
def tenth(x): return 0.1*x
def Sign(x): return np.sign(x)
def Pi(): return np.pi
def pi(x): return x*np.pi
def sin(x): return np.sin(x)
def const_1(): return 1.0
def const_5(): return 5.0
def const_01(): return 0.1
def const_002(): return 0.02

new_functions = [
    Function(Add, 2, name='Add'),
    Function(Sub, 2, name='Sub'),
    Function(Mul, 2, name='Mul'),
    Function(Div, 2, name='Div'),
    Function(Max1, 1, name='Max2'),
    Function(Sum, 1, name='Sum'),
    # Function(Min1, 1, name='Min'),
    Function(Sign, 1, name='Sgn'),
    Function(const_01, 0, name='0.1')
]

fs = [
        # Function(const_1, 0),
        # Function(const_5, 0),
        # Function(const_01, 0),
        Function(const_002, 0),

        # Function(op.add, 2), 
        Function(op.sub, 2), 
        # Function(op.mul, 2), 
        Function(protected_div, 2),
        Function(op.neg, 1),
        Function(op.abs, 1),
        # Function(op.ge, 2),
        # Function(op.le, 2),
        
        # Function(op.pow, 2),
        # Function(exp, 1),
        # Function(max1, 1),
        # Function(min1, 1),
        # Function(max2, 2),
        # Function(min2, 2),
        # Function(tenth, 1),
        # Function(scaled, 1),
        Function(sign, 1),
        # Function(uniform, 2),
        # Function(uniform01, 0),
        # Function(relu, 1),
        # Function(sin, 1),
        # Function(pi,1)
        # Function(ln, 1),
        # Function(sqrt, 1),
        # Function(inv, 1),
    ]

# Map Python functions to sympy counterparts for symbolic simplification.
DEFAULT_SYMBOLIC_FUNCTION_MAP = {
    'const_1':              const_1,
    'const_5':              const_5,
    'const_01':             const_01,
    'const_002':            const_002,
    # op.and_.__name__:       sp.And,
    # op.or_.__name__:        sp.Or,
    # op.not_.__name__:       sp.Not,
    op.add.__name__:        op.add,
    op.sub.__name__:        op.sub,
    op.mul.__name__:        op.mul,
    'protected_div':        op.truediv,
    op.neg.__name__:        op.neg,
    'abs':                  op.abs,
    'max1':                 max1,
    'min1':                 min1,
    'max2':                 max2,
    'min2':                 min2,
    'tenth':                tenth,
    'scaled':               scaled,
    'sign':                 sign,
    
    # op.pow.__name__:        op.pow,
    # op.abs.__name__:        op.abs,
    # op.floordiv.__name__:   op.floordiv,
    # op.truediv.__name__:    op.truediv,
    
    # math.log.__name__:      sp.log,
    # math.sin.__name__:      sp.sin,
    # math.cos.__name__:      sp.cos,
    # math.tan.__name__:      sp.tan,
    
}


# LunarLander专用函数库
LUNARLANDER_SYMBOLIC_FUNCTION_MAP = {
    # op.and_.__name__:       sp.And,
    # op.or_.__name__:        sp.Or,
    # op.not_.__name__:       sp.Not,
    op.add.__name__:        op.add,
    op.sub.__name__:        op.sub,
    op.mul.__name__:        op.mul,
    op.neg.__name__:        op.neg,
    'max1':                 max1,
    'tenth':                tenth,
    'scaled':               scaled,
    'sign':                 sign,
    'abs':                  op.abs,
    # op.pow.__name__:        op.pow,
    # op.abs.__name__:        op.abs,
    # op.floordiv.__name__:   op.floordiv,
    # op.truediv.__name__:    op.truediv,
    # 'protected_div':        op.truediv,
    # math.log.__name__:      sp.log,
    # math.sin.__name__:      sp.sin,
    # math.cos.__name__:      sp.cos,
    # math.tan.__name__:      sp.tan,
    # 'const_1':              const_1,
    # 'const_5':              const_5,
    # 'const_tenth':          const_01,
}
if __name__ == '__main__':
    
    pass