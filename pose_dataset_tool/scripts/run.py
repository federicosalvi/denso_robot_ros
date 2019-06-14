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

camera_params = rospy.wait_for_message("/camera/camera_info", CameraInfo)
tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
bridge = CvBridge()
rospy.sleep(2)


mesh = MeshPly('/root/catkin_ws/bracket.ply')
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()

group.set_max_acceleration_scaling_factor(0.05)
group.set_max_velocity_scaling_factor(0.15)

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

rr, ss, tt = generate_meshgrid(7)


dir = '/tmp/pictures/'
#for file in os.listdir(dir):
#  os.remove(dir + file)

rotation = 0
j = 0
while not rospy.is_shutdown():
    offset = [1,0,0]
    target_faces = get_faces(mesh, rotation=rotation, offset=offset)
    target_corners = get_3D_corners(vertices, rotation=rotation, offset=offset)
    target_width, target_length, target_height = get_box_size(vertices)
    target_centroid = Point(offset[0], offset[1], offset[2] + target_height/2)
    target_points = np.hstack((np.vstack((vector_from_point(target_centroid),[1])),target_corners))

    target_pose = place_target_in_scene(target_centroid, (target_width, target_length, target_height), rotation, scene, rospy)

    i = 0
    sphere_origin = target_centroid
    while not rospy.is_shutdown():
        if i == len(rr):
            break
        pose.position = sample_sphere(rr[i], ss[i], tt[i]) # , offset=sphere_origin)
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
        if not success:
            actual = group.get_current_pose().pose.position
            difference = point_difference(pose.position,actual)
            if difference > 1e-02:
                rospy.logerr('\nFailed for position:\n{}\n'.format(pose))
                i += 1
                continue

        rospy.sleep(2)
        #rospy.loginfo('\nMoved to position:\n{}\njoints values:\n{}\n'.format(pose,group.get_current_joint_values()))

        frame_transform, object_pose = transform_pose(tf_buffer, rospy, target_pose.pose)
        #rospy.loginfo('\nFrame transform:\n{}\n'.format(frame_transform))
        #rospy.loginfo('\nObject center transform:\n{}\n'.format(object_pose))
        rotation_matrix = rotm_from_quaternion(frame_transform.transform.rotation)
        translation_vector = vector_from_point(frame_transform.transform.translation)
        #rospy.loginfo('\nTransform:\n\trotation matrix:\n{}\n\ttranslation vector:\n{}\n'.format(rotation_matrix, translation_vector))

        # project 3d points onto image plane
        proj_points = proj_to_camera(target_points, rotation_matrix, translation_vector, camera_params).T
        proj_faces = []
        for face in target_faces:
            projected_face = []
            proj_faces.append(proj_to_camera(face, rotation_matrix, translation_vector, camera_params).T)
        proj_faces = np.array(proj_faces, dtype=int)
        save_seg_mask('{}{:04d}.png'.format(dir,j), proj_faces, rospy, bridge)
         # take picture and write labels
        take_picture('{}{:06d}.jpg'.format(dir,j), bridge, rospy)
        f = open('{}{:06d}.txt'.format(dir,j),'w+')
        output = '42 ' # class label

        # 2D coordinates of centroid and 8 corners
        for x,y in proj_points:
            output += str(x/camera_params.width) + ' ' + str(y/camera_params.height) + ' '

        # xrange and yrange
        output += str((np.max(proj_points[:,0])-np.min(proj_points[:,0]))/camera_params.width) + ' ' + str((np.max(proj_points[:,1])-np.min(proj_points[:,1]))/camera_params.height)
        f.write(output)
        f.close()

        i += 1
        j += 1
    rotation += np.pi/2
    if np.abs(rotation - (np.pi*2)) < 1e-2:
        rospy.loginfo('Finished')
        break
    raw_input('Rotate the object 90 degrees counter-clockwise and press enter:')
    
moveit_commander.roscpp_shutdown()
