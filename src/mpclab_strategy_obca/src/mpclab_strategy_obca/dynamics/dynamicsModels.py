#!/usr/bin python3

# from casadi import *
import casadi as ca
import numpy as np

from mpclab_strategy_obca.dynamics.utils.types import dynamicsKinBikeParams

class bike_dynamics_rk4(object):
	"""docstring for bike_dynamics_rk4"""
	def __init__(self, params=dynamicsKinBikeParams()):
		self.L_f = params.L_f
		self.L_r = params.L_r

		self.dt = params.dt

		self.M = params.M

	def f_ct(self, x, u, type='casadi'):
		if type == 'casadi':
			beta = lambda d: ca.atan2(self.L_r * ca.tan(d), self.L_r + self.L_f)

			x_dot = vcat([ x[3]*ca.cos(x[2] + ca.beta(u[0])),
							x[3]*ca.sin(x[2] + beta(u[0])),
							x[3]*ca.sin(beta(u[0])) / self.L_r,
							u[1] ])
		elif type == 'numpy':
			beta = lambda d: np.atan2(self.L_r * np.tan(d), self.L_r + self.L_f)

			x_dot = np.array([x[3]*np.cos(x[2] + np.beta(u[0])),
							x[3]*np.sin(x[2] + beta(u[0])),
							x[3]*np.sin(beta(u[0])) / self.L_r,
							u[1]])
		else:
			raise RuntimeError('Dynamics type %s not recognized' % type)
			
		return x_dot

	def f_dt(self, x_k, u_k, type='casadi'):

		h_k = self.dt / self.M

		# Runge-Kutta to obtain discrete time dynamics
		x_kp1 = x_k

		for i in range(self.M):
			a1 = self.f_ct(x_kp1, u_k, type)
			a2 = self.f_ct(x_kp1+h_k*a1/2, u_k, type)
			a3 = self.f_ct(x_kp1+h_k*a2/2, u_k, type)
			a4 = self.f_ct(x_kp1+h_k*a3, u_k, type)
			x_kp1 = x_kp1 + h_k*(a1 + 2*a2 + 2*a3 + a4)/6

		return x_kp1

	def f_dt_aug(self, x_k, u_k):
		x_kp1 = ca.vcat([self.f_dt(x_k, u_k), u_k])

		return x_kp1
