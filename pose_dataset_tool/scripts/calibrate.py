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
rospy.init_node('vs_setup_arm',anonymous=True)
scene = PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("arm")
robot = moveit_commander.RobotCommander()

rospy.sleep(2)

rotation = 0
offset = [1,0,0]

target_centroid = Point(offset[0], offset[1], offset[2])

group.set_max_acceleration_scaling_factor(0.05)
group.set_max_velocity_scaling_factor(0.05)

CAMERA_UPRIGHT = np.pi-0.79
group.set_goal_tolerance(0.0001)
print(group.get_goal_tolerance())
#group.set_joint_value_target({
#  "joint_1": 0,
#  "joint_2": 0,
#  "joint_3": np.pi/2,
#  "joint_4": 0,
#  "joint_5": np.pi/2,
#  "joint_6": CAMERA_UPRIGHT
#})
# dummy initial pose
if len(sys.argv) < 2:
        rotation = 0
else:
        rotation = float(sys.argv[1])/180 * np.pi


orient = Quaternion(*tf.transformations.quaternion_from_euler(0,np.pi,rotation))
pose = Pose(Point( 0.7, 0, 0.1), orient)
group.set_pose_target(pose)
success = group.go(True)
group.stop()
print(group.get_current_joint_values())


#while not rospy.is_shutdown():
#	pose.position = sample_sphere(rr[i], ss[i], tt[i])
#        forward = normalize([
#		target_centroid.x - pose.position.x,
#		target_centroid.y - pose.position.y,
#		target_centroid.z - pose.position.z
#	])
#	pose.orientation = look_at(forward, rotate=False)

#	group.set_pose_target(pose)
#  	success = group.go(True, wait=True)
#        group.stop()
#        group.clear_pose_targets()
#	rospy.sleep(2)

#	i += 1
#	if i == len(rr):
#		i = 0

moveit_commander.roscpp_shutdown()
