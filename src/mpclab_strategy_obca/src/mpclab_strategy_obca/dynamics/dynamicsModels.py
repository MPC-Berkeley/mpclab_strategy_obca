#!/usr/bin python3

from casadi import *

from mpclab_strategy_obca.dynamics.utils.dynamicsTypes import dynamicsKinBikeParams

class bike_dynamics_rk4(object):
	"""docstring for bike_dynamics_rk4"""
	def __init__(self, params=dynamicsKinBikeParams()):
		self.L_f = params.L_f
		self.L_r = params.L_r

		self.dt = params.dt

		self.M = params.M

	def f_ct(self, x, u):
		beta = lambda d: atan2(self.L_r * tan(d), self.L_r + self.L_f)

		x_dot = vcat([ x[3]*cos(x[2] + beta(u[0])),
						x[3]*sin(x[2] + beta(u[0])),
						x[3]*sin(beta(u[0])) / self.L_r, 
						u[1] ])

		return x_dot

	def f_dt(self, x_k, u_k):
		
		h_k = self.dt / self.M

		# Runge-Kutta to obtain discrete time dynamics
		x_kp1 = x_k

		for i in range(self.M):
			a1 = self.f_ct(x_kp1, u_k)
			a2 = self.f_ct(x_kp1+h_k*a1/2, u_k)
			a3 = self.f_ct(x_kp1+h_k*a2/2, u_k)
			a4 = self.f_ct(x_kp1+h_k*a3, u_k)
			x_kp1 = x_kp1 + h_k*(a1 + 2*a2 + 2*a3 + a4)/6

		return x_kp1

	def f_dt_aug(self, x_k, u_k):
		x_kp1 = vcat([self.f_dt(x_k, u_k), u_k])

		return x_kp1
