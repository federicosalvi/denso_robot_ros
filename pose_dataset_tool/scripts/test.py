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
camera = image_geometry.PinholeCameraModel()
camera.fromCameraInfo(camera_params)

tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
bridge = CvBridge()
pub = rospy.Publisher('/camera/camera_with_points', Image, queue_size=10)
pub_seg = rospy.Publisher('/camera/camera_with_segm', Image, queue_size=10)

camera = image_geometry.PinholeCameraModel()
camera.fromCameraInfo(camera_params)

rospy.sleep(2)

rotation = 0
offset = [1,0,0.24]

mesh = MeshPly('/root/catkin_ws/bracket.ply')
target_faces = get_faces(mesh, rotation=rotation, offset=offset)
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
target_corners = get_3D_corners(vertices, rotation=rotation, offset=offset)
target_width, target_length, target_height = get_box_size(vertices)
target_centroid = Point(offset[0], offset[1], offset[2]+target_height/2)
target_points = np.hstack((np.vstack((vector_from_point(target_centroid),[1])),target_corners))
# shape of the points is [4, npoints]

target_pose = place_target_in_scene(target_centroid, (target_width, target_length, target_height), rotation, scene, rospy)
#target2_centroid = Point(offset[0], offset[1]-0.09, offset[2]+target_height/2)
#place_target_in_scene(target2_centroid, (target_width, target_length, target_height), rotation, scene, rospy, 'target2')

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

group.set_path_constraints(add_robot_constraints())

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
	rospy.sleep(2)

	#rospy.loginfo('\nMoved to position:\n{}\njoints values:\n{}\n'.format(pose,group.get_current_joint_values()))

	frame_transform, object_pose = transform_pose(tf_buffer, rospy, target_pose.pose)
	rospy.loginfo('\nFrame transform:\n{}\n'.format(frame_transform))
	rospy.loginfo('\nObject center transform:\n{}\n'.format(object_pose))
	rotation_matrix = rotm_from_quaternion(frame_transform.transform.rotation)
	translation_vector = vector_from_point(frame_transform.transform.translation)
	rospy.loginfo('\nTransform:\n\trotation matrix:\n{}\n\ttranslation vector:\n{}\n'.format(rotation_matrix, translation_vector))

	# transform point from world to camera frame
        rt = np.concatenate((rotation_matrix, translation_vector), axis=1)
        transformed_target_points = rt.dot(target_points).T

	# project 3d points onto image plane
	proj_points = []
	for point in transformed_target_points:
		proj_points.append([int(x) for x in camera.project3dToPixel(tuple(point))])
	proj_points = np.array(proj_points)
	publish_image_with_points(proj_points, rospy, bridge, pub)

	# transform faces from world to camera frame
	transformed_target_faces = []
	for face in target_faces:
		transformed_target_faces.append(rt.dot(face).T)
	transformed_target_faces = np.array(transformed_target_faces)

	# project faces from world to camera frame
	proj_faces = []
	for face in transformed_target_faces:
		proj_face = []
		for point in face:
			proj_face.append([int(x) for x in camera.project3dToPixel(tuple(point))])
		proj_faces.append(proj_face)
	proj_faces = np.array(proj_faces, dtype=int)
	overlay_seg_mask(proj_faces, rospy, bridge, pub_seg)

	i += 1
	if i == len(rr):
		i = 0

moveit_commander.roscpp_shutdown()
