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

camera_params = rospy.wait_for_message("/camera/camera_info", CameraInfo)
tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
bridge = CvBridge()
pub = rospy.Publisher('/camera/camera_with_points', Image, queue_size=10)
rospy.sleep(2)

rotation = 0

mesh = MeshPly('/root/catkin_ws/bracket.ply')
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
target_corners = get_3D_corners(vertices, rotation=rotation, offset=[1,0,0])
target_width, target_length, target_height = get_box_size(vertices)
target_centroid = Point(1, 0, target_height/2)
target_points = np.hstack((np.vstack((vector_from_point(target_centroid),[1])),target_corners))

target_pose = place_target_in_scene(target_centroid, (target_width, target_length, target_height), rotation, scene, rospy)

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


poses = [
	Point(0,-0.5,0.5),
	Point(0,-0.5,0.4),
	Point(0,-0.5,0.3),
	Point(0,-0.5,0.2),
	Point(0,-0.5,0.1),
	Point(0,-0.5,0.09),
	Point(0,-0.5,0.08),
	Point(0,-0.5,0.07),
	Point(0,-0.5,0.06),
	Point(0,-0.5,0.05)
]
i = 0
while not rospy.is_shutdown():
	pose.position = poses[i]
	pose.orientation = Quaternion(
		*tf.transformations.quaternion_from_euler(0,np.pi/2,0))

	group.set_pose_target(pose)
   	success = group.go(True, wait=True)
        group.stop()
        group.clear_pose_targets()
	rospy.sleep(2)

	rospy.loginfo('\nMoved to position:\n{}\njoints values:\n{}\n'.format(pose,group.get_current_joint_values()))

	frame_transform, object_pose = transform_pose(tf_buffer, rospy, target_pose.pose)
	rospy.loginfo('\nFrame transform:\n{}\n'.format(frame_transform))
	rospy.loginfo('\nObject center transform:\n{}\n'.format(object_pose))
	rotation_matrix = rotm_from_quaternion(frame_transform.transform.rotation)
	translation_vector = vector_from_point(frame_transform.transform.translation)
	rospy.loginfo('\nTransform:\n\trotation matrix:\n{}\n\ttranslation vector:\n{}\n'.format(rotation_matrix, translation_vector))

	# project 3d points onto image plane
	#transformed_target_corners = transform_points(target_corners)
	#proj_points = proj_to_camera(target_points, rotation_matrix, translation_vector, camera_params).T
	#rospy.loginfo('Projected points:\n{}'.format(proj_points))

	#publish_image_with_points(np.array(proj_points), rospy, bridge, pub)
	i += 1
	if i == len(poses):
		break
	raw_input('Press enter for the next pose')

moveit_commander.roscpp_shutdown()
