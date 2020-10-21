#!/usr/bin/env python3
"""
	File name: plotCar.py
	Author: Xu Shen
	Email: xu_shen@berkeley.edu
	Python Version: 2.7
"""
import rospy

from barc.utils.envs import OvalTrack, CircleTrack, LabTrack, LTrack, ParkingLane

from mpclab_strategy_obca.visualization.visualizer import barcOBCAVisualizer
from mpclab_strategy_obca.visualization.types import visualizerParams, plotterParams

def main():
	rospy.init_node('obca_visualizer')

	track_type = rospy.get_param('/track/shape')
	plot_subplots = rospy.get_param('/visualization/plot_subplots', True)
	plot_sim      = rospy.get_param('/visualization/plot_sim', True)
	plot_est      = rospy.get_param('/visualization/plot_est', True)

	namespaces = rospy.get_param('/visualization/namespaces')
	colors = rospy.get_param('/visualization/colors')
	trajectory_file = rospy.get_param('/target_vehicle/controller/trajectory_file', None)
	scaling_factor = rospy.get_param('/target_vehicle/controller/scaling_factor', 1.0)

	dt = rospy.get_param('/visualization/dt')
	loop_rate = 1.0/dt
	rate = rospy.Rate(loop_rate)

	if track_type == "oval":
		track = OvalTrack(track_width=0.8)
	elif track_type == "circle":
		track = CircleTrack(track_width=0.8)
	elif track_type == "LabTrack":
		track = LabTrack(init_pos=(0, 0, 0))
	elif track_type == "LTrack":
		track = LTrack()
	elif track_type == "ParkingLane":
		track = ParkingLane()
	else:
		raise ValueError('Chosen Track shape not valid.')

	vis_params = visualizerParams(dt=dt, plot_subplots=plot_subplots,
		plot_sim=plot_sim, plot_est=plot_est,
		trajectory_file=trajectory_file, scaling_factor=scaling_factor)
	vis = barcOBCAVisualizer(track, vis_params)
	for (n, c) in zip(namespaces, colors):
		d = {'car_width' : rospy.get_param(n + '/car/plot/W'), 'car_length' : rospy.get_param(n + '/car/plot/L')}
		plot_state = rospy.get_param(n + '/controller/plot_state')
		plot_score = rospy.get_param(n + '/controller/plot_score')
		plot_pred    = rospy.get_param(n + '/controller/plot_pred')
		global_pred    = rospy.get_param(n + '/controller/global_pred')
		p = plotterParams(plot_state=plot_state, plot_score=plot_score,
			plot_pred=plot_pred, global_pred=global_pred)
		vis.attach_plotter(n, c, d, p)

	rospy.sleep(1.0)

	while not rospy.is_shutdown():
		# Update the plot
		vis.update()

		rate.sleep()

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
