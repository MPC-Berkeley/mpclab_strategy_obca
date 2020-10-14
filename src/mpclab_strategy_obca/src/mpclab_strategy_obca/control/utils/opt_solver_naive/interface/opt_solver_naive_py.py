#opt_solver_naive : A fast customized optimization solver.
#
#Copyright (C) 2013-2020 EMBOTECH AG [info@embotech.com]. All rights reserved.
#
#
#This software is intended for simulation and testing purposes only.
#Use of this software for any commercial purpose is prohibited.
#
#This program is distributed in the hope that it will be useful.
#EMBOTECH makes NO WARRANTIES with respect to the use of the software
#without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
#PARTICULAR PURPOSE.
#
#EMBOTECH shall not have any liability for any damage arising from the use
#of the software.
#
#This Agreement shall exclusively be governed by and interpreted in
#accordance with the laws of Switzerland, excluding its principles
#of conflict of laws. The Courts of Zurich-City shall have exclusive
#jurisdiction in case of any dispute.
#
#def __init__():
'''
a Python wrapper for a fast solver generated by FORCES PRO v4.0.0

   OUTPUT = opt_solver_naive_py.opt_solver_naive_solve(PARAMS) solves a multistage problem
   subject to the parameters supplied in the following dictionary:
       PARAMS['x0'] - column vector of length 542
       PARAMS['xinit'] - column vector of length 6
       PARAMS['all_parameters'] - column vector of length 462

   OUTPUT returns the values of the last iteration of the solver where
       OUTPUT['x01'] - column vector of size 26
       OUTPUT['x02'] - column vector of size 26
       OUTPUT['x03'] - column vector of size 26
       OUTPUT['x04'] - column vector of size 26
       OUTPUT['x05'] - column vector of size 26
       OUTPUT['x06'] - column vector of size 26
       OUTPUT['x07'] - column vector of size 26
       OUTPUT['x08'] - column vector of size 26
       OUTPUT['x09'] - column vector of size 26
       OUTPUT['x10'] - column vector of size 26
       OUTPUT['x11'] - column vector of size 26
       OUTPUT['x12'] - column vector of size 26
       OUTPUT['x13'] - column vector of size 26
       OUTPUT['x14'] - column vector of size 26
       OUTPUT['x15'] - column vector of size 26
       OUTPUT['x16'] - column vector of size 26
       OUTPUT['x17'] - column vector of size 26
       OUTPUT['x18'] - column vector of size 26
       OUTPUT['x19'] - column vector of size 26
       OUTPUT['x20'] - column vector of size 26
       OUTPUT['x21'] - column vector of size 22

   [OUTPUT, EXITFLAG] = opt_solver_naive_py.opt_solver_naive_solve(PARAMS) returns additionally
   the integer EXITFLAG indicating the state of the solution with
       1 - Optimal solution has been found (subject to desired accuracy)
       2 - (only branch-and-bound) A feasible point has been identified for which the objective value is no more than codeoptions.mip.mipgap*100 per cent worse than the global optimum
       0 - Timeout - maximum number of iterations reached
      -1 - (only branch-and-bound) Infeasible problem (problems solving the root relaxation to the desired accuracy)
      -2 - (only branch-and-bound) Out of memory - cannot fit branch and bound nodes into pre-allocated memory.
      -6 - NaN or INF occured during evaluation of functions and derivatives. Please check your initial guess.
      -7 - Method could not progress. Problem may be infeasible. Run FORCESdiagnostics on your problem to check for most common errors in the formulation.
     -10 - The convex solver could not proceed due to an internal error
    -100 - License error

   [OUTPUT, EXITFLAG, INFO] = opt_solver_naive_py.opt_solver_naive_solve(PARAMS) returns
   additional information about the last iterate:
       INFO.it        - number of iterations that lead to this result
       INFO.it2opt    - number of convex solves
       INFO.res_eq    - max. equality constraint residual
       INFO.res_ineq  - max. inequality constraint residual
       INFO.rsnorm    - norm of stationarity condition
       INFO.rcompnorm    - max of all complementarity violations
       INFO.pobj      - primal objective
       INFO.mu        - duality measure
       INFO.solvetime - Time needed for solve (wall clock time)
       INFO.fevalstime - Time needed for function evaluations (wall clock time)

 See also COPYING

'''

import ctypes
import os
import numpy as np
import numpy.ctypeslib as npct
import sys

#_lib = ctypes.CDLL(os.path.join(os.getcwd(),'opt_solver_naive/lib/opt_solver_naive.so'))
try:
	_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../lib/opt_solver_naive_withModel.so'))
	csolver = getattr(_lib,'opt_solver_naive_solve')
except:
	_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../lib/libopt_solver_naive_withModel.so'))
	csolver = getattr(_lib,'opt_solver_naive_solve')

class opt_solver_naive_params_ctypes(ctypes.Structure):
#	@classmethod
#	def from_param(self):
#		return self
	_fields_ = [('x0', ctypes.c_double * 542),
('xinit', ctypes.c_double * 6),
('all_parameters', ctypes.c_double * 462),
]

opt_solver_naive_params = {'x0' : np.array([]),
'xinit' : np.array([]),
'all_parameters' : np.array([]),
}
params = {'x0' : np.array([]),
'xinit' : np.array([]),
'all_parameters' : np.array([]),
}
opt_solver_naive_params_types = {'x0' : np.float64,
'xinit' : np.float64,
'all_parameters' : np.float64,
}

class opt_solver_naive_outputs_ctypes(ctypes.Structure):
#	@classmethod
#	def from_param(self):
#		return self
	_fields_ = [('x01', ctypes.c_double * 26),
('x02', ctypes.c_double * 26),
('x03', ctypes.c_double * 26),
('x04', ctypes.c_double * 26),
('x05', ctypes.c_double * 26),
('x06', ctypes.c_double * 26),
('x07', ctypes.c_double * 26),
('x08', ctypes.c_double * 26),
('x09', ctypes.c_double * 26),
('x10', ctypes.c_double * 26),
('x11', ctypes.c_double * 26),
('x12', ctypes.c_double * 26),
('x13', ctypes.c_double * 26),
('x14', ctypes.c_double * 26),
('x15', ctypes.c_double * 26),
('x16', ctypes.c_double * 26),
('x17', ctypes.c_double * 26),
('x18', ctypes.c_double * 26),
('x19', ctypes.c_double * 26),
('x20', ctypes.c_double * 26),
('x21', ctypes.c_double * 22),
]

opt_solver_naive_outputs = {'x01' : np.array([]),
'x02' : np.array([]),
'x03' : np.array([]),
'x04' : np.array([]),
'x05' : np.array([]),
'x06' : np.array([]),
'x07' : np.array([]),
'x08' : np.array([]),
'x09' : np.array([]),
'x10' : np.array([]),
'x11' : np.array([]),
'x12' : np.array([]),
'x13' : np.array([]),
'x14' : np.array([]),
'x15' : np.array([]),
'x16' : np.array([]),
'x17' : np.array([]),
'x18' : np.array([]),
'x19' : np.array([]),
'x20' : np.array([]),
'x21' : np.array([]),
}


class opt_solver_naive_info(ctypes.Structure):
#	@classmethod
#	def from_param(self):
#		return self
	_fields_ = [('it', ctypes.c_int),
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

class FILE(ctypes.Structure):
        pass
if sys.version_info.major == 2:
	PyFile_AsFile = ctypes.pythonapi.PyFile_AsFile # problem here with python 3 http://stackoverflow.com/questions/16130268/python-3-replacement-for-pyfile-asfile
	PyFile_AsFile.argtypes = [ctypes.py_object]
	PyFile_AsFile.restype = ctypes.POINTER(FILE)

# determine data types for solver function prototype
csolver.argtypes = ( ctypes.POINTER(opt_solver_naive_params_ctypes), ctypes.POINTER(opt_solver_naive_outputs_ctypes), ctypes.POINTER(opt_solver_naive_info), ctypes.POINTER(FILE))
csolver.restype = ctypes.c_int

def opt_solver_naive_solve(params_arg):
	'''
a Python wrapper for a fast solver generated by FORCES PRO v4.0.0

   OUTPUT = opt_solver_naive_py.opt_solver_naive_solve(PARAMS) solves a multistage problem
   subject to the parameters supplied in the following dictionary:
       PARAMS['x0'] - column vector of length 542
       PARAMS['xinit'] - column vector of length 6
       PARAMS['all_parameters'] - column vector of length 462

   OUTPUT returns the values of the last iteration of the solver where
       OUTPUT['x01'] - column vector of size 26
       OUTPUT['x02'] - column vector of size 26
       OUTPUT['x03'] - column vector of size 26
       OUTPUT['x04'] - column vector of size 26
       OUTPUT['x05'] - column vector of size 26
       OUTPUT['x06'] - column vector of size 26
       OUTPUT['x07'] - column vector of size 26
       OUTPUT['x08'] - column vector of size 26
       OUTPUT['x09'] - column vector of size 26
       OUTPUT['x10'] - column vector of size 26
       OUTPUT['x11'] - column vector of size 26
       OUTPUT['x12'] - column vector of size 26
       OUTPUT['x13'] - column vector of size 26
       OUTPUT['x14'] - column vector of size 26
       OUTPUT['x15'] - column vector of size 26
       OUTPUT['x16'] - column vector of size 26
       OUTPUT['x17'] - column vector of size 26
       OUTPUT['x18'] - column vector of size 26
       OUTPUT['x19'] - column vector of size 26
       OUTPUT['x20'] - column vector of size 26
       OUTPUT['x21'] - column vector of size 22

   [OUTPUT, EXITFLAG] = opt_solver_naive_py.opt_solver_naive_solve(PARAMS) returns additionally
   the integer EXITFLAG indicating the state of the solution with
       1 - Optimal solution has been found (subject to desired accuracy)
       2 - (only branch-and-bound) A feasible point has been identified for which the objective value is no more than codeoptions.mip.mipgap*100 per cent worse than the global optimum
       0 - Timeout - maximum number of iterations reached
      -1 - (only branch-and-bound) Infeasible problem (problems solving the root relaxation to the desired accuracy)
      -2 - (only branch-and-bound) Out of memory - cannot fit branch and bound nodes into pre-allocated memory.
      -6 - NaN or INF occured during evaluation of functions and derivatives. Please check your initial guess.
      -7 - Method could not progress. Problem may be infeasible. Run FORCESdiagnostics on your problem to check for most common errors in the formulation.
     -10 - The convex solver could not proceed due to an internal error
    -100 - License error

   [OUTPUT, EXITFLAG, INFO] = opt_solver_naive_py.opt_solver_naive_solve(PARAMS) returns
   additional information about the last iterate:
       INFO.it        - number of iterations that lead to this result
       INFO.it2opt    - number of convex solves
       INFO.res_eq    - max. equality constraint residual
       INFO.res_ineq  - max. inequality constraint residual
       INFO.rsnorm    - norm of stationarity condition
       INFO.rcompnorm    - max of all complementarity violations
       INFO.pobj      - primal objective
       INFO.mu        - duality measure
       INFO.solvetime - Time needed for solve (wall clock time)
       INFO.fevalstime - Time needed for function evaluations (wall clock time)

 See also COPYING

	'''
	global _lib

	# convert parameters
	params_py = opt_solver_naive_params_ctypes()
	for par in params_arg:
		try:
			#setattr(params_py, par, npct.as_ctypes(np.reshape(params_arg[par],np.size(params_arg[par]),order='A')))
			params_arg[par] = np.require(params_arg[par], dtype=opt_solver_naive_params_types[par], requirements='F')
			setattr(params_py, par, npct.as_ctypes(np.reshape(params_arg[par],np.size(params_arg[par]),order='F')))
		except:
			raise ValueError('Parameter ' + par + ' does not have the appropriate dimensions or data type. Please use numpy arrays for parameters.')

	outputs_py = opt_solver_naive_outputs_ctypes()
	info_py = opt_solver_naive_info()
	if sys.version_info.major == 2:
		if sys.platform.startswith('win'):
			fp = None # if set to none, the solver prints to stdout by default - necessary because we have an access violation otherwise under windows
		else:
			#fp = open('stdout_temp.txt','w')
			fp = sys.stdout
		try:
			PyFile_AsFile.restype = ctypes.POINTER(FILE)
			exitflag = _lib.opt_solver_naive_solve( params_py, ctypes.byref(outputs_py), ctypes.byref(info_py), PyFile_AsFile(fp) , _lib.opt_solver_naive_casadi2forces )
			#fp = open('stdout_temp.txt','r')
			#print (fp.read())
			#fp.close()
		except:
			#print 'Problem with solver'
			raise
	elif sys.version_info.major == 3:
		if sys.platform.startswith('win'):
			libc = ctypes.cdll.msvcrt
		elif sys.platform.startswith('darwin'):
			libc = ctypes.CDLL('libc.dylib')
		else:
			libc = ctypes.CDLL('libc.so.6')       # Open libc
		cfopen = getattr(libc,'fopen')        # Get its fopen
		cfopen.restype = ctypes.POINTER(FILE) # Yes, fopen gives a file pointer
		cfopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p] # Yes, fopen gives a file pointer
		fp = cfopen('stdout_temp.txt'.encode('utf-8'),'w'.encode('utf-8'))    # Use that fopen

		try:
			if sys.platform.startswith('win'):
				exitflag = _lib.opt_solver_naive_solve( params_py, ctypes.byref(outputs_py), ctypes.byref(info_py), None , _lib.opt_solver_naive_casadi2forces)
			else:
				exitflag = _lib.opt_solver_naive_solve( params_py, ctypes.byref(outputs_py), ctypes.byref(info_py), fp , _lib.opt_solver_naive_casadi2forces)
			libc.fclose(fp)
			fptemp = open('stdout_temp.txt','r')
			print (fptemp.read())
			fptemp.close()
		except:
			#print 'Problem with solver'
			raise

	# convert outputs
	for out in opt_solver_naive_outputs:
		opt_solver_naive_outputs[out] = npct.as_array(getattr(outputs_py,out))

	return opt_solver_naive_outputs,int(exitflag),info_py

solve = opt_solver_naive_solve