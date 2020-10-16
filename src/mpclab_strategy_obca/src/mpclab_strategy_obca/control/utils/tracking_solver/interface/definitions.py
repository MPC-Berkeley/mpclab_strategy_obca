import numpy
import ctypes

name = "tracking_solver"
requires_callback = True
lib = "lib/libtracking_solver.so"
lib_static = "lib/libtracking_solver.a"
c_header = "include/tracking_solver.h"

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("lb"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (398,   1),  398),
 ("ub"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (398,   1),  398),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (404,   1),  404),
 ("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  6,   1),    6),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (608,   1),  608)]

# Output                | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x02"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x03"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x04"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x05"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x06"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x07"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x08"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x09"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x10"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x11"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x12"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x13"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x14"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x15"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x16"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x17"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x18"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x19"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x20"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x21"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x22"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x23"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x24"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x25"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x26"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x27"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x28"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x29"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x30"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x31"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x32"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x33"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x34"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x35"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x36"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x37"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x38"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x39"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x40"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x41"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x42"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x43"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x44"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x45"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x46"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x47"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x48"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x49"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x50"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  8,),    8),
 ("x51"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  4,),    4)]

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