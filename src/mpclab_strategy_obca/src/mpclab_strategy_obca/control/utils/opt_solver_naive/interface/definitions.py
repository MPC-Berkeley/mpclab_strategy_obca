import numpy
import ctypes

name = "opt_solver_naive"
requires_callback = True
lib = "lib/libopt_solver_naive.so"
lib_static = "lib/libopt_solver_naive.a"
c_header = "include/opt_solver_naive.h"

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (542,   1),  542),
 ("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  6,   1),    6),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (462,   1),  462)]

# Output                | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x02"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x03"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x04"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x05"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x06"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x07"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x08"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x09"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x10"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x11"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x12"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x13"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x14"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x15"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x16"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x17"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x18"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x19"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x20"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 26,),   26),
 ("x21"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 22,),   22)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
('it2opt', ctypes.c_int),
('res_eq', ctypes.c_double),
('res_ineq', ctypes.c_double),
('rsnorm', ctypes.c_double),
('rcompnorm', ctypes.c_double),
('pobj', ctypes.c_double),
('dobj', ctypes.c_double),
('dgap', ctypes.c_double),
('rdgap', ctypes.c_double),
('mu', ctypes.c_double),
('mu_aff', ctypes.c_double),
('sigma', ctypes.c_double),
('lsit_aff', ctypes.c_int),
('lsit_cc', ctypes.c_int),
('step_aff', ctypes.c_double),
('step_cc', ctypes.c_double),
('solvetime', ctypes.c_double),
('fevalstime', ctypes.c_double)
]