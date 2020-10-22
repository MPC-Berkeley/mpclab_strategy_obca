#!/usr/bin/env python3
"""
	File name: plotCar.py
	Author: Xu Shen
	Email: xu_shen@berkeley.edu
	Python Version: 2.7
"""
import rospy
import numpy as np

from barc.utils.envs import OvalTrack, CircleTrack, LabTrack, LTrack, ParkingLane

from mpclab_strategy_obca.visualization.visualizer import barcOBCAVisualizer
from mpclab_strategy_obca.visualization.types import visualizerParams, plotterParams

def main():
	rospy.init_node('obca_visualizer')

	track_type = rospy.get_param('/track/shape')
	num_parking_spots = rospy.get_param('/track/num_parking_spots')
	parking_spot_width = rospy.get_param('/track/parking_spot_width')

	plot_subplots = rospy.get_param('/visualization/plot_subplots', True)
	plot_sim      = rospy.get_param('/visualization/plot_sim', True)
	plot_est      = rospy.get_param('/visualization/plot_est', True)

	namespaces = rospy.get_param('/visualization/namespaces')
	colors = rospy.get_param('/visualization/colors')
	trajectory_file = rospy.get_param('/target_vehicle/controller/trajectory_file', None)
	x_scaling = rospy.get_param('/target_vehicle/controller/x_scaling', 1.0)
	y_scaling = rospy.get_param('/target_vehicle/controller/y_scaling', 1.0)
	v_scaling = rospy.get_param('/target_vehicle/controller/v_scaling', 1.0)
	x_init = rospy.get_param('/target_vehicle/car/car_init/x', 0.0)
	y_init = rospy.get_param('/target_vehicle/car/car_init/y', 0.0)
	heading_init = rospy.get_param('/target_vehicle/car/car_init/heading', 0.0)
	if type(heading_init) is str:
		heading_init = eval(heading_init)
	v_init = rospy.get_param('/target_vehicle/car/car_init/v', 0.0)
	trajectory_scaling = {'x': x_scaling, 'y': y_scaling, 'v': v_scaling}
	trajectory_init = {'x': x_init, 'y': y_init, 'heading': heading_init, 'v': v_init}

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
		track_length = rospy.get_param('/track/length')
		track = ParkingLane(init_pos=(0, 0, 0), length=track_length)
	else:
		raise ValueError('Chosen Track shape not valid.')

	vis_params = visualizerParams(dt=dt, plot_subplots=plot_subplots,
		plot_sim=plot_sim, plot_est=plot_est,
		trajectory_file=trajectory_file, trajectory_scaling=trajectory_scaling, trajectory_init=trajectory_init,
		parking_spot_width=parking_spot_width, num_parking_spots=num_parking_spots)
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
