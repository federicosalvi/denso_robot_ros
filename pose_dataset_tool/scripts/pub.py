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

camera_params = load_camera_parameters()
tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
bridge = CvBridge()

rotation = 0
mesh = MeshPly('/root/catkin_ws/bracket.ply')
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
target_corners = get_3D_corners(vertices, rotation=rotation, offset=[1,0,0])
target_width, target_length, target_height = get_box_size(vertices)
target_centroid = Point(1, 0, target_height/2)
target_points = np.hstack((np.vstack((vector_from_point(target_centroid),[1])),target_corners))

target_pose = Pose()
target_pose.position = target_centroid
target_pose.orientation =  Quaternion(*tf.transformations.quaternion_from_euler(0,0,rotation))


rate = rospy.Rate(1)
while not rospy.is_shutdown():
	frame_transform, object_pose = transform_pose(tf_buffer, rospy, target_pose)
	rotation_matrix = rotm_from_quaternion(frame_transform.transform.rotation)
	translation_vector = vector_from_point(frame_transform.transform.translation)
	proj_points = proj_to_camera(target_points, rotation_matrix, translation_vector, camera_params).T
	publish_image_with_points(np.array(proj_points), rospy, bridge, pub)
	rate.sleep()
