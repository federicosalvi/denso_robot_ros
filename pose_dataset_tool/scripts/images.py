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

rospy.init_node('tf_pub',anonymous=True)

bridge = CvBridge()

rate = rospy.Rate(1)
filepath = '/tmp/pictures/image_'
i = 0
while not rospy.is_shutdown():
	take_picture('{}{}.png'.format(filepath,i), bridge, rospy)
	i += 1
	rate.sleep()
