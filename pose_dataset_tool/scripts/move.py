#!/usr/bin/env python
import sys, rospy, tf, moveit_commander, random, tf2_ros
import tf2_geometry_msgs
from moveit_commander.planning_scene_interface import PlanningSceneInterface
#from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, PointStamped
from moveit_msgs.msg import OrientationConstraint, JointConstraint, VisibilityConstraint, Constraints
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import image_geometry
import os
from MeshPly import MeshPly
from library import *

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('vs_move_arm',anonymous=True)
scene = PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("arm")
robot = moveit_commander.RobotCommander()

rospy.sleep(2)

target_centroid = Point(1, -.1, 0)

group.set_max_acceleration_scaling_factor(0.05)
group.set_max_velocity_scaling_factor(0.05)

CAMERA_UPRIGHT = np.pi-0.79

group.set_joint_value_target({
  "joint_1": 0,
  "joint_2": 0,
  "joint_3": np.pi/2,
  "joint_4": 0,
  "joint_5": np.pi/2,
  "joint_6": CAMERA_UPRIGHT
})
# dummy initial pose
orient = Quaternion(*tf.transformations.quaternion_from_euler(0,np.pi/4,0))
pose = Pose(Point( 0.7, 0, 0.5), orient)
#group.set_pose_target(pose)
success = group.go(True)
group.stop()


rr, ss, tt = generate_meshgrid(5)
i = 0
while not rospy.is_shutdown():

	pose.position = sample_sphere(rr[i], ss[i], tt[i])
        forward = normalize([
		target_centroid.x - pose.position.x,
		target_centroid.y - pose.position.y,
		target_centroid.z - pose.position.z
	])
	pose.orientation = look_at(forward, rotate=True)

	group.set_pose_target(pose)
   	success = group.go(True, wait=True)
        group.stop()
        group.clear_pose_targets()
	i+=1
	q = raw_input('Press enter for the next pose, q to quit')
	if q == 'q':
		break

moveit_commander.roscpp_shutdown()
