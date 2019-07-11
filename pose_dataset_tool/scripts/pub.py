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

rospy.init_node('vs_tf_pub',anonymous=True)

camera_params = rospy.wait_for_message('/camera/camera_info', CameraInfo)
camera = image_geometry.PinholeCameraModel()
camera.fromCameraInfo(camera_params)
tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
bridge = CvBridge()
pub = rospy.Publisher('/camera/camera_with_points', Image, queue_size=10)
pub_seg = rospy.Publisher('/camera/camera_with_segm', Image, queue_size=10)
if len(sys.argv) < 4:
	print('Usage: rosrun pose_dataset_tool pub.py <model filename> <rotation in degrees> <height offset>')
	exit()
rotation = float(sys.argv[2])/180 * np.pi
height_offset = float(sys.argv[3])

mesh = MeshPly('/root/catkin_ws/src/denso_robot_ros/pose_dataset_tool/models/{}'.format(sys.argv[1]))
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
target_width, target_length, target_height = get_box_size(vertices)
offset=[1,0,target_height/2 + height_offset]
target_faces = get_faces(mesh, rotation=rotation, offset=offset)
target_corners = get_3D_corners(vertices, rotation=rotation, offset=offset)
target_centroid = Point(offset[0], offset[1], offset[2]+target_height/2)
target_points = np.hstack((np.vstack((vector_from_point(target_centroid),[1])),target_corners))

target_pose = Pose()
target_pose.position = target_centroid
target_pose.orientation =  Quaternion(*tf.transformations.quaternion_from_euler(0,0,rotation))


rate = rospy.Rate(1)
while not rospy.is_shutdown():
	frame_transform, object_pose = transform_pose(tf_buffer, rospy, target_pose)
	rotation_matrix = rotm_from_quaternion(frame_transform.transform.rotation)
	translation_vector = vector_from_point(frame_transform.transform.translation)
#	new_points = list(camera_params.K)
#	new_points[2] -= 1
#	new_points[5] -= 1
#	camera_params.K = tuple(new_points)
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

	rate.sleep()
