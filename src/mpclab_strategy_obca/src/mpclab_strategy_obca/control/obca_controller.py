#!/usr/bin python3

import numpy as np
from scipy.linalg import block_diag, sqrtm
import casadi as ca
import forcespro
import forcespro.nlp

from mpclab_strategy_obca.control.abstractController import abstractController

from mpclab_strategy_obca.control.utils.controllerTypes import strategyOBCAParams

from mpclab_strategy_obca.dynamics.dynamicsModels import bike_dynamics_rk4

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

	def solve_ws(self, z, u, obs):
		pass

	def generate_ws_solver(self):
		ws_model = forcespro.nlp.SymbolicModel()

		ws_model.N = self.N + 1

		ws_model.objective = self.eval_ws_obj
		ws_model.objectiveN = lambda z: 0

		# [lambda, mu, d]
		ws_model.nvar = self.N_ineq + self.M_ineq + self.n_obs
		ws_model.ub = np.hstack( [ float('inf') * np.ones(self.N_ineq + self.M_ineq), 
									float('inf') * np.ones(self.n_obs) ] )
		ws_model.lb = np.hstack( [np.zeros(self.N_ineq + self.M_ineq), 
									-float('inf')*np.ones(self.n_obs)] )

		# [obca dist, obca]
		ws_model.neq = self.n_obs + self.n_obs*self.d_ineq
		ws_model.eq = self.eval_ws_eq
		# ws_model.E = 0.1*np.ones((ws_model.neq, ws_model.nvar))
		ws_model.E = np.hstack( [np.eye(ws_model.neq), np.zeros((ws_model.neq, ws_model.nvar-ws_model.neq))] )

		# [obca norm]
		ws_model.nh = self.n_obs
		ws_model.ineq = self.eval_ws_ineq
		ws_model.hu = np.ones(self.n_obs)
		ws_model.hl = -float('inf')*np.ones(self.n_obs)

		# [x_ref, obs_A, obs_b]
		ws_model.npar = self.n_x + self.N_ineq*self.d_ineq + self.N_ineq

		ws_codeopts = forcespro.CodeOptions(self.ws_name)

		ws_codeopts.overwrite = 1
		ws_codeopts.printlevel = 1
		ws_codeopts.optlevel = self.optlevel
		ws_codeopts.BuildSimulinkBlock = 0
		ws_codeopts.dump_formulation = 0
		ws_codeopts.nlp.linear_solver = 'symm_indefinite'
		ws_codeopts.nlp.ad_tool = 'casadi-3.5.1'

		return ws_model.generate_solver(ws_codeopts)


	def eval_ws_obj(self, z):

		d = z[self.N_ineq + self.M_ineq:] # d = z(N_ineq+M_ineq+1:end)

		return -ca.sum1(d)

	def eval_ws_eq(self, z, p):
		
		t_ws = p[0:2]
		R_ws = np.array( [ [ca.cos(p[2]), -ca.sin(p[2])],
							[ca.sin(p[2]), ca.cos(p[2])] ] )

		obca_d = []
		obca = []

		j = 0

		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )
			b = p[ self.n_x + self.N_ineq*self.d_ineq + j : self.n_x + self.N_ineq*self.d_ineq + j + self.n_ineq[i] ]

			Lambda = z[ j : j + self.n_ineq[i] ]

			mu = z[ self.N_ineq + (i-1)*self.m_ineq : self.N_ineq + i*self.m_ineq ]

			d = z[ self.N_ineq + self.M_ineq + i ]

			obca_d.append( -ca.dot(self.g, mu) + ca.dot( ca.mtimes(A, t_ws)-b, Lambda ) - d)

			obca.append( ca.mtimes(self.G.T, mu) + ca.mtimes( ca.mtimes(A, R_ws).T, Lambda ) )

			j += self.n_ineq[i]

		ws_eq = ca.vcat( [ca.vcat(obca_d),
							ca.vcat(obca)] )

		return ws_eq

	def eval_ws_ineq(self, z, p):
		
		ws_ineq = []

		j = 0
		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )

			Lambda = z[ j : j + self.n_ineq[i] ]

			ws_ineq.append( ca.dot( ca.mtimes(A.T, Lambda), ca.mtimes(A.T, Lambda) ) )

			j += self.n_ineq[i]

		return ca.vcat( ws_ineq )

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
			# ls_objective.append( self.eval_opt_ls_obj )

			# [x_k, lambda_k, mu_k, u_k, u_km1]
			nvar.append( self.n_x + self.N_ineq + self.M_ineq + self.n_u + self.n_u )
			ub.append( np.hstack( [float('inf')*np.ones(self.n_x), 
									float('inf')*np.ones(self.N_ineq+self.M_ineq), 
									self.u_u, 
									self.u_u] ) )
			lb.append( np.hstack( [-float('inf')*np.ones(self.n_x), 
									np.ones(self.N_ineq+self.M_ineq), 
									self.u_l, 
									self.u_l] ) )

			# [obca, obca_d, obca_norm, hyp, du]
			nh.append( self.n_obs*self.d_ineq + self.n_obs + self.n_obs + 1 + self.n_u )
			ineq.append( self.eval_opt_ineq )
			hu.append( np.hstack( [1e-8*np.ones(self.n_obs*self.d_ineq), 
									float('inf')*np.ones(self.n_obs), 
									np.ones(self.n_obs), 
									float('inf'), 
									self.dt*self.du_u] ) )
			hl.append( np.hstack( [np.zeros(self.n_obs*self.d_ineq), 
									self.d_min*np.ones(self.n_obs), 
									-float('inf')*np.ones(self.n_obs), 
									0, 
									self.dt*self.du_l] ) )

			# [x_ref, obs_A, obs_b, hyp_w, hyp_b]
			npar.append( self.n_x + self.N_ineq*self.d_ineq + self.N_ineq + self.n_x + 1 )
			if i == opt_model.N - 2:
				# [dynamics, obca]
				# neq.append( self.n_x + self.n_obs*self.d_ineq )
				neq.append(self.n_x)
				E.append( np.hstack( [np.eye(self.n_x), np.zeros((self.n_x, self.N_ineq+self.M_ineq))] ) )
				eq.append( self.eval_opt_eq_Nm1 )
			else:
				# [augmented dynamics, obca]
				# neq.append( self.n_x + self.n_u + self.n_obs*self.d_ineq )
				neq.append( self.n_x + self.n_u )
				E.append( np.block( [ [np.eye(self.n_x), np.zeros((self.n_x, self.N_ineq + self.M_ineq + self.n_u + self.n_u))], 
										[np.zeros((self.n_u, self.n_x+self.N_ineq+self.M_ineq+self.n_u)), np.eye(self.n_u)] ] ) )
				eq.append( self.eval_opt_eq )


		objective.append( self.eval_opt_obj_N )
		# ls_objective.append( self.eval_opt_ls_obj_N )

		# [x_k, lambda_k, mu_k]
		nvar.append( self.n_x + self.N_ineq + self.M_ineq )
		ub.append( np.hstack( [float('inf')*np.ones(self.n_x), 
								float('inf')*np.ones(self.N_ineq + self.M_ineq)] ) )
		lb.append( np.hstack( [-float('inf')*np.ones(self.n_x), 
								np.zeros(self.N_ineq + self.M_ineq)] ) )

		# [obca, obca_d, obca_norm, hyp]
		nh.append( self.n_obs*self.d_ineq + self.n_obs + self.n_obs + 1 )
		ineq.append( self.eval_opt_ineq_N )
		hu.append( np.hstack( [1e-8*np.ones(self.n_obs*self.d_ineq),
								float('inf')*np.ones(self.n_obs), 
								np.ones(self.n_obs),
								float('inf')] ) )
		hl.append( np.hstack( [np.zeros(self.n_obs*self.d_ineq),
								self.d_min*np.ones(self.n_obs), 
								-float('inf')*np.ones(self.n_obs), 
								0] ) )

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
		print('nh', nh)
		print('hu', [t.shape for t in hu])
		print('hl', [t.shape for t in hl])

		opt_model.xinitidx = np.hstack((np.arange(self.n_x), 
										np.arange(self.n_x+self.N_ineq+self.M_ineq+self.n_u, self.n_x+self.N_ineq+self.M_ineq+self.n_u+self.n_u)))

		opt_codeopts = forcespro.CodeOptions(self.opt_name)
		opt_codeopts.overwrite = 1
		opt_codeopts.printlevel = 2
		opt_codeopts.optlevel = self.optlevel
		opt_codeopts.BuildSimulinkBlock = 0
		opt_codeopts.dump_formulation = 0

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

		opt_obj = ca.bilin(self.Q, x-x_ref, x-x_ref ) + ca.bilin( self.R, u, u)

		return opt_obj


	def eval_opt_obj_N(self, z, p):
		x = z[:self.n_x]
		x_ref = p[:self.n_x]

		opt_obj = ca.bilin( self.Q, x-x_ref, x-x_ref )

		return opt_obj

	def eval_opt_eq(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		return self.dynamics.f_dt_aug(x, u)

	def eval_opt_eq_Nm1(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		return self.dynamics.f_dt(x, u)

	def eval_opt_ineq(self, z, p):
		x = z[:self.n_x]
		u = z[self.n_x+self.N_ineq+self.M_ineq : self.n_x+self.N_ineq+self.M_ineq+self.n_u]

		u_p = z[self.n_x+self.N_ineq+self.M_ineq+self.n_u : self.n_x+self.N_ineq+self.M_ineq+self.n_u+self.n_u]
		hyp_w = p[self.n_x+self.N_ineq*self.d_ineq+self.N_ineq : self.n_x+self.N_ineq*self.d_ineq+self.N_ineq+self.n_x]
		hyp_b = p[self.n_x+self.N_ineq*self.d_ineq+self.N_ineq+self.n_x]

		t_opt = z[0:2]
		R_opt = np.array([ [ca.cos(z[2]), -ca.sin(z[2])], [ca.sin(z[2]), ca.cos(z[2])] ])

		obca = []
		obca_d = []
		obca_norm = []

		j = 0

		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )
			b = p[ self.n_x + self.N_ineq*self.d_ineq + j : self.n_x + self.N_ineq*self.d_ineq + j + self.n_ineq[i] ]

			Lambda = z[ self.n_x + j : self.n_x + j + self.n_ineq[i] ]

			mu = z[ self.n_x + self.N_ineq + (i-1)*self.m_ineq : self.n_x + self.N_ineq + i*self.m_ineq ]

			obca.append( ca.mtimes( ca.transpose(self.G), mu) + ca.mtimes( ca.transpose(ca.mtimes(A, R_opt)), Lambda ) )
			obca_d.append( -ca.dot(self.g, mu) + ca.dot(ca.mtimes(A, t_opt)-b, Lambda) )
			obca_norm.append( ca.dot(ca.mtimes( ca.transpose(A), Lambda), ca.mtimes( ca.transpose(A), Lambda)) )

			j += self.n_ineq[i]

		opt_ineq = ca.vcat( [ca.vcat(obca), ca.vcat(obca_d), ca.vcat(obca_norm)] )
		opt_ineq = ca.vcat( [opt_ineq, ca.dot(hyp_w, x) - hyp_b] )
		opt_ineq = ca.vcat( [opt_ineq, u-u_p] )

		return opt_ineq

	def eval_opt_ineq_N(self, z, p):
		x = z[:self.n_x]

		hyp_w = p[self.n_x+self.N_ineq*self.d_ineq+self.N_ineq : self.n_x+self.N_ineq*self.d_ineq+self.N_ineq+self.n_x]
		hyp_b = p[self.n_x+self.N_ineq*self.d_ineq+self.N_ineq+self.n_x]

		t_opt = z[0:2]
		R_opt = np.array([ [ca.cos(z[2]), -ca.sin(z[2])], [ca.sin(z[2]), ca.cos(z[2])] ])

		obca = []
		obca_d = []
		obca_norm = []

		j = 0

		for i in range(self.n_obs):
			A = p[ self.n_x + j*self.d_ineq : self.n_x + (j+self.n_ineq[i])*self.d_ineq ].reshape( (self.n_ineq[i], self.d_ineq) )
			b = p[ self.n_x + self.N_ineq*self.d_ineq + j : self.n_x + self.N_ineq*self.d_ineq + j + self.n_ineq[i] ]

			Lambda = z[ self.n_x + j : self.n_x + j + self.n_ineq[i] ]

			mu = z[ self.n_x + self.N_ineq + (i-1)*self.m_ineq : self.n_x + self.N_ineq + i*self.m_ineq ]

			obca.append( ca.mtimes( ca.transpose(self.G), mu) + ca.mtimes( ca.transpose(ca.mtimes(A, R_opt)), Lambda ) )
			obca_d.append( -ca.dot(self.g, mu) + ca.dot(ca.mtimes(A, t_opt)-b, Lambda) )
			obca_norm.append( ca.dot(ca.mtimes( ca.transpose(A), Lambda), ca.mtimes( ca.transpose(A), Lambda)) )

			j += self.n_ineq[i]

		opt_ineq = ca.vcat( [ca.vcat(obca), ca.vcat(obca_d), ca.vcat(obca_norm)] )
		opt_ineq = ca.vcat( [opt_ineq, ca.dot(hyp_w, x) - hyp_b] )

		return opt_ineq


def main():
	dynamics = bike_dynamics_rk4()
	controller = obca_controller(dynamics)
	controller.initialize(regen=True)

if __name__ == '__main__':
	main()