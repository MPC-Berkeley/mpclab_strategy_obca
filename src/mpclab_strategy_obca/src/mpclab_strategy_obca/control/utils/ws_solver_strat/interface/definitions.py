import numpy
import ctypes

name = "ws_solver_strat"
requires_callback = True
lib = "lib/libws_solver_strat.so"
lib_static = "lib/libws_solver_strat.a"
c_header = "include/ws_solver_strat.h"

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (441,   1),  441),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (462,   1),  462)]

# Output                | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x02"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x03"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x04"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x05"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x06"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x07"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x08"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x09"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x10"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x11"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x12"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x13"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x14"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x15"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x16"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x17"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x18"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x19"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x20"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21),
 ("x21"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 21,),   21)]

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