#!/usr/bin python3

import numpy as np
from scipy.linalg import block_diag, sqrtm
import casadi
import forcespro
import forcespro.nlp

from mpclab_strategy_obca.control.abstractController import abstractController

from mpclab_strategy_obca.control.utils.controllerTypes import strategyOBCAParams

class obca_controller(abstractController):
	"""docstring for ClassName"""
	def __init__(self, dynamics, params=strategyOBCAParams()):

		self.dynamics = dynamics

		self.dt = params.dt

		self.n_x = params.n
		self.n_u = params.d
		
		self.N = params.N
		
		self.n_obs = params.n_obs
		self.n_ineq = params.n_ineq # Number of constraints for each obstacle
		self.d_ineq = params.d_ineq # Dimension of constraints for all obstacles

		self.G = params.G
		self.g = params.g
		self.m_ineq = self.G.shape[0] # Number of constraints for controlled object

		self.Q = params.Q
		self.R = params.R

		self.d_min = params.d_min

		self.u_l = params.u_l
		self.u_u = params.u_u
		self.du_l = params.du_l
		self.du_u = params.du_u

		self.N_ineq = np.sum(self.n_ineq)
		self.M_ineq = self.n_obs * self.m_ineq

		self.optlevel = params.optlevel

		self.z_ws = None
		self.u_ws = None
		self.lambda_ws = None
		self.mu_ws = None

		self.ws_name = params.ws_name
		self.opt_name = params.opt_name

		self.ws_solver = None
		self.opt_solver = None

	def initialize(self, regen):
		
		if regen:
			self.ws_solver = self.generate_ws_solver()
			self.opt_solver = self.generate_opt_solver()
		else:
			self.ws_solver = forcespro.nlp.Solver.from_directory('/path/to/solver')
			self.opt_solver = forcespro.nlp.Solver.from_directory('/path/to/solver')

	def solve(self):
		pass

	def generate_ws_solver(self):
		ws_model = forcespro.nlp.SymbolicModel()

		ws_model.N = self.N + 1

		ws_model.objective = self.eval_ws_obj
		ws_model.objectiveN = lambda z: 0

		# [lambda, mu, d]
		ws_model.nvar = self.N_ineq + self.M_ineq + self.n_obs;
		ws_model.ub = np.concatenate( [ float('inf') * np.ones((self.N_ineq + self.M_ineq, 1)), 
										float('inf') * np.ones((self.n_obs, 1)) ], 
										axis=0 )
		ws_model.lb = np.concatenate( [ np.zeros((self.N_ineq + self.M_ineq, 1)),
										-float('inf') * np.ones((self.n_obs, 1)) ],
										axis = 0 )

		# [obca dist, obca]
		ws_model.neq = self.n_obs + self.n_obs * self.d_ineq
		ws_model.eq = self.eval_ws_eq;
		ws_model.E = zeros(ws_model.neq, ws_model.nvar);

		# [obca norm]
		ws_model.nh = self.n_obs;
		ws_model.ineq = self.eval_ws_ineq;
		ws_model.hu = np.ones((self.n_obs,1));
		ws_model.hl = -float('inf')*np.ones((self.n_obs,1));

		# [x_ref, obs_A, obs_b]
		ws_model.npar = self.n_x + self.N_ineq*self.d_ineq + self.N_ineq;

		ws_codeopts = forcespro.CodeOptions(self.ws_name)

		ws_codeopts.overwrite = 1;
		ws_codeopts.printlevel = 1;
		ws_codeopts.optlevel = self.optlevel;
		ws_codeopts.BuildSimulinkBlock = 0;
		ws_codeopts.nlp.linear_solver = 'symm_indefinite';
		ws_codeopts.nlp.ad_tool = 'casadi-3.5.1';

		return ws_model.generate_solver(ws_codeopts)


	def eval_ws_obj(self, z):

		d = z[self.N_ineq + self.M_ineq:] # d = z(N_ineq+M_ineq+1:end);

		return -np.sum(d)

	def eval_ws_eq(self, z, p):
		
		t_ws = p[0:1]
		R_ws = np.array( [ [np.cos(p[2]), -np.sin(p[2])],
							[np.sin(p[2]), np.cos(p[2])] ] )

		obca_d = []
		obca = []

		j = 0

		for i in range(n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )
			b = p[ self.n_x + self.N_ineq*self.d_ineq + j : self.n_x + self.N_ineq*self.d_ineq + j + self.n_ineq[i] ]

			Lambda = z[ j : j + self.n_ineq[i] ]

			mu = z[ self.N_ineq + (i-1)*self.m_ineq : self.N_ineq + i*self.m_ineq ]

			d = z[ self.N_ineq + self.M_ineq + i ]

			obca_d.append( -np.dot(self.g, mu) + np.dot( np.matmul(A, t_ws)-b, Lambda ) - d)

			obca.append( np.matmul(self.G.T, mu) + np.matmul( np.matmul(A, R_ws).T, Lambda ) )

			j += self.n_ineq[i]

		ws_eq = np.concatenate( [np.concatenate(obca_d, axis=0),
								 np.concatenate(obca, axis=0)],
								 axis = 0 )


		return ws_eq

	def eval_ws_ineq(self, z, p):
		
		ws_ineq = []

		j = 0
		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )

			Lambda = z[ j : j + self.n_ineq[i] ]

			ws_ineq.append( np.dot( np.matmul(A.T, Lambda), np.matmul(A.T, Lambda) ) )

			j += self.n_ineq[i]

		return np.concatenate( ws_ineq, axis=0 )

	def generate_opt_solver(self):
		opt_model = forcespro.nlp.SymbolicModel()

		opt_model.N = self.N + 1

		nvar         = []
		neq          = []
		nh           = []
		npar         = []
		objective    = []
		ls_objective = []
		eq           = []
		E            = []
		ineq         = []
		hu           = []
		hl           = []
		ub           = []
		lb           = []

		for i in range(opt_model.N-1):
			objective.append( self.eval_opt_obj )
			ls_objective.append( self.eval_opt_ls_obj )

			# [x_k, lambda_k, mu_k, u_k, u_km1]
			nvar.append( self.n_x + self.N_ineq + self.M_ineq + self.n_u + self.n_u )
			ub.append( [np.float('inf')*np.ones((self.n_x, 1)), 
						np.float('inf')*np.ones((self.N_ineq+self.M_ineq, 1)), 
						self.u_u, 
						self.u_u] )
			lb.append( [-np.float('inf')*np.ones((self.n_x, 1)), 
						np.ones((self.n_obs, 1)), 
						self.u_l, 
						self.u_l] )

			# [obca_d, obca_norm, hyp, du]
			nh.append( self.n_obs + self.n_obs + 1 + self.n_u )
			ineq.append( self.eval_opt_ineq )
			hu.append( [np.float('inf')*np.ones((self.n_obs, 1)), 
						np.ones((self.n_obs, 1)), 
						np.float('inf'), 
						self.dt*self.du_u] )
			hl.append( [self.d_min*np.ones((self.n_obs, 1)), 
						-np.float('inf')*np.ones((self.n_obs, 1)), 
						0, 
						self.dt*self.du_l] )

			# [x_ref, obs_A, obs_b, hyp_w, hyp_b]
			npar.append( self.n_x + self.N_ineq*self.d_ineq + self.N_ineq + self.n_x + 1 )

			if i == opt_model.N - 2:
				# [dynamics, obca]
				neq.append( self.n_x + self.n_obs*self.d_ineq )
				E.append( np.block( [ [np.eye(self.n_x), np.zeros((self.n_x, self.N_ineq+self.M_ineq))], 
										[np.zeros((self.n_obs*self.d_ineq, self.n_x + self.N_ineq + self.M_ineq)) ] ] ) )
				eq.append( self.eval_opt_eq_Nm1 )
			else:
				# [augmented dynamics, obca]
				neq.append( self.n_x + self.n_obs*self.d_ineq )
				E.append( np.block( [ [np.eye(self.n_x), np.zeros((self.n_x, self.N_ineq+self.M_ineq))], 
										[np.zeros((self.n_u, self.n_x+self.N_ineq+self.M_ineq+self.n_u)), self.eye(self.n_u)], 
										[np.zeros((self.n_obs*self.d_ineq, self.n_x+self.N_ineq+self.M_ineq+self.n_u+self.n_u))] ] ) )
				eq.append( self.eval_opt_eq )

		objective.append( self.eval_opt_obj_N )
		ls_objective.append( self.eval_opt_ls_obj_N )

		# [x_k, lambda_k, mu_k]
		nvar.append( self.n_x + self.N_ineq + self.M_ineq )
		ub.append( [np.float('inf')*np.ones((self.n_x, 1)), 
					np.float('inf')*np.ones((self.N_ineq + self.M_ineq, 1))] )
		lb.append( [-np.float('inf')*np.ones((self.n_x, 1)), 
					np.zeros((self.N_ineq + self.M_ineq, 1))] )

		# [obca_d, obca_norm, hyp]
		nh.append( self.n_obs + self.n_obs + 1 )
		ineq.append( self.eval_opt_ineq_N )
		hu.append( [np.float('inf')*np.ones((self.n_obs, 1)), 
					np.ones((self.n_obs, 1)),
					np.float('inf')] )
		hl.append( [self.d_min*np.ones((self.n_obs, 1)), 
					-np.float('inf')*np.ones((self.n_obs, 1))] )

		# [x_ref, obs_A, obs_b, hyp_w, hyp_b]
		npar.append( self.n_x + self.N_ineq*self.d_ineq + self.N_ineq +self.n_x + 1 )


		opt_model.nvar = nvar
		opt_model.neq = neq
		opt_model.nh = nh
		opt_model.npar = npar

		opt_model.objective = objective
		opt_model.LSobjective = ls_objective
		opt_model.eq = eq
		opt_model.E = E
		opt_model.ineq = ineq
		opt_model.hu = hu
		opt_model.hl = hl
		opt_model.ub = ub
		opt_model.lb = lb

		opt_model.xinitidx = np.hstack((np.arange(0:self.n_x), 
										np.arange(self.n_x+self.N_ineq+self.M_ineq+self.n_u, self.n_x+self.N_ineq+self.M_ineq+self.n_u+self.n_u)))

		opt_codeopts = forcespro.CodeOptions(self.opt_name)
		opt_codeopts.overwrite = 1
		opt_codeopts.printlevel = 2
		opt_codeopts.optlevel = self.optlevel
		opt_codeopts.BuildSimulinkBlock = 0

		opt_codeopts.nlp.ad_tool = 'casadi-3.5.1'
		opt_codeopts.nlp.linear_solver = 'symm_indefinite'

		opt_codeopts.nlp.TolStat = 1e-3
		opt_codeopts.nlp.TolEq = 1e-3
		opt_codeopts.nlp.TolIneq = 1e-3
		opt_codeopts.accuracy.ineq = 1e-3  # infinity norm of residual for inequalities
		opt_codeopts.accuracy.eq = 1e-3    # infinity norm of residual for equalities
		opt_codeopts.accuracy.mu = 1e-3    # absolute duality gap
		opt_codeopts.accuracy.rdgap = 1e-3

		return opt_model.generate_solver(opt_codeopts)

	def eval_opt_obj(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		x_ref = p[:self.n_x]

		opt_obj = np.matmul( np.matmul(x-x_ref, self.Q), x-x_ref ) + np.matmul( np.matmul(u, R), u )

		return opt_obj

	def eval_opt_ls_obj(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		x_ref = p[:self.n_x]

		opt_obj = np.matmul( sqrtm(block_diag(self.Q, self.R)), np.vstack((x-x_ref, u)) )

		return opt_obj

	def eval_opt_obj_N(self, z, p):
		x = z[:self.n_x]
		x_ref = p[:self.n_x]

		opt_obj = np.matmul( np.matmul(x-x_ref, self.Q), x-x_ref )

		return opt_obj

	def eval_opt_ls_obj_N(self, z, p):
		x = z[:self.n_x]
		x_ref = p[:self.n_x]

		opt_obj = np.matmul( sqrtm(self.Q), x-x_ref)

		return opt_obj

	def eval_opt_eq(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		R_opt = np.array([ [np.cos(z[2]), -np.sin(z[2])], [np.sin(z[2]), np.cos(z[2])] ])

		j = 0

		opt_eq = self.dynamics.f_dt_aug(x, u)

		obca = []

		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )

			Lambda = z[ self.n_x + j : self.n_x + j + self.n_ineq[i] ]

			mu = z[ self.n_x + self.N_ineq + (i-1)*self.m_ineq : self.n_x + self.N_ineq + i*self.m_ineq ]

			obca.append( np.matmul(self.G.T, mu) + np.matmul( np.matmul(A, R_opt).T, Lambda ) )

			j += self.n_ineq[i]

		opt_eq = np.concatenate(obca, axis=0)

		return opt_eq

	def eval_opt_eq_Nm1(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		R_opt = np.array([ [np.cos(z[2]), -np.sin(z[2])], [np.sin(z[2]), np.cos(z[2])] ])

		j = 0

		opt_eq = self.dynamics.f_dt(x, u)

		obca = []

		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )

			Lambda = z[ self.n_x + j : self.n_x + j + self.n_ineq[i] ]

			mu = z[ self.n_x + self.N_ineq + (i-1)*self.m_ineq : self.n_x + self.N_ineq + i*self.m_ineq ]

			obca.append( np.matmul(self.G.T, mu) + np.matmul( np.matmul(A, R_opt).T, Lambda ) )

			j += self.n_ineq[i]

		opt_eq = np.concatenate(obca, axis=0)

		return opt_eq

	def eval_opt_ineq(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		u_p = z[self.n_x+self.N_ineq+self.M_ineq+self.n_u : self.n_x+self.N_ineq+self.M_ineq+self.n_u+self.n_u]
		hyp_w = p[self.n_x+self.N_ineq*self.d_ineq+self.N_ineq : self.n_x+self.N_ineq*self.d_ineq+self.N_ineq+self.n_x]
		hyp_b = p[self.n_x+self.N_ineq*self.d_ineq+self.N_ineq+self.n_x];

		t_opt = z[0:2]

		obca_d = []
		obca_norm = []

		j = 0

		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )
			b = p[ self.n_x + self.N_ineq*self.d_ineq + j : self.n_x + self.N_ineq*self.d_ineq + j + self.n_ineq[i] ]

			Lambda = z[ self.n_x + j : self.n_x + j + self.n_ineq[i] ]

			mu = z[ self.n_x + self.N_ineq + (i-1)*self.m_ineq : self.n_x + self.N_ineq + i*self.m_ineq ]

			obca.append( -np.dot(self.g, mu) + np.dot(np.matmul(A, t_opt)-b, Lambda) )
			obca_norm.append( np.dot(np.matmul(A.T, Lambda), np.matmul(A.T, Lambda)) )

			j += self.n_ineq[i]

		opt_ineq = np.vstack( (np.concatenate(obca, axis=0), np.concatenate(obca_norm, axis=0)) )
		opt_ineq = np.vstack( (opt_ineq, np.dot(hyp_w, x) - hyp_b) )
		opt_ineq = np.vstack( (opt_ineq, u-u_p) )

		return opt_ineq

	def eval_opt_ineq_N(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		u_p = z[self.n_x+self.N_ineq+self.M_ineq+self.n_u : self.n_x+self.N_ineq+self.M_ineq+self.n_u+self.n_u]
		hyp_w = p[self.n_x+self.N_ineq*self.d_ineq+self.N_ineq : self.n_x+self.N_ineq*self.d_ineq+self.N_ineq+self.n_x]
		hyp_b = p[self.n_x+self.N_ineq*self.d_ineq+self.N_ineq+self.n_x];

		t_opt = z[0:2]

		obca_d = []
		obca_norm = []

		j = 0

		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )
			b = p[ self.n_x + self.N_ineq*self.d_ineq + j : self.n_x + self.N_ineq*self.d_ineq + j + self.n_ineq[i] ]

			Lambda = z[ self.n_x + j : self.n_x + j + self.n_ineq[i] ]

			mu = z[ self.n_x + self.N_ineq + (i-1)*self.m_ineq : self.n_x + self.N_ineq + i*self.m_ineq ]

			obca.append( -np.dot(self.g, mu) + np.dot(np.matmul(A, t_opt)-b, Lambda) )
			obca_norm.append( np.dot(np.matmul(A.T, Lambda), np.matmul(A.T, Lambda)) )

			j += self.n_ineq[i]

		opt_ineq = np.vstack( (np.concatenate(obca, axis=0), np.concatenate(obca_norm, axis=0)) )
		opt_ineq = np.vstack( (opt_ineq, np.dot(hyp_w, x) - hyp_b) )

		return opt_ineq
