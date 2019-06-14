#!/usr/bin/env python
import sys, tf, random, tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, PointStamped
from moveit_msgs.msg import OrientationConstraint, JointConstraint, VisibilityConstraint, Constraints
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import image_geometry

def normalize(v):
    v = np.array(v)
    return v/np.linalg.norm(v)

def sample_sphere(r, s, t):
    x = r * np.cos(t) * np.sin(s)
    y = r * np.sin(t) * np.sin(s)
    z = r * np.cos(s)
    return Point(x,y,z)

def look_at(forward, rotate=False):
    camera_up = np.array([0,0,1])
    left = normalize(np.cross(camera_up, forward))
    up = np.cross(forward, left)

    # we're switching the axis since the camera "points" along the z-axis
    R = np.vstack((-left, -up, forward)).T
    if rotate:
	R = rotate_random(R)

    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return Quaternion(*tf.transformations.quaternion_from_euler(x,y,z))

def transform_pose(tf_buffer, rospy, pose, target_frame='camera', source_frame='world'):
  targetPose = PoseStamped()
  targetPose.header.stamp = rospy.Time.now()
  targetPose.header.frame_id = source_frame
  targetPose.pose = pose
  rospy.sleep(2)

  transform = tf_buffer.lookup_transform_full(target_frame, rospy.Time(), source_frame, rospy.Time(), 'world', rospy.Duration(10.0))
  transformed_pose = tf2_geometry_msgs.do_transform_pose(targetPose, transform).pose
  # wait until the camera is in position
#  while np.abs(transformed_pose.position.y) > 0.001 and not rospy.is_shutdown():
 #   transform = tf_buffer.lookup_transform_full(target_frame, rospy.Time().now(), source_frame, rospy.Time(), 'world', rospy.Duration(10.0))
  #  transformed_pose = tf2_geometry_msgs.do_transform_pose(targetPose, transform).pose
  return transform, transformed_pose

def transform_points(tf_buffer, rospy, points, target_frame='camera', source_frame='world'):
  # shape of the input points should be (npoints, 3) or (npoints, 4) in case of homogeneous coordinates
  transformed_points = np.empty((points.shape[0], 3))
  for i,point in enumerate(points):
    targetPoint = PointStamped()
    targetPoint.header.stamp = rospy.Time.now()
    targetPoint.header.frame_id = source_frame
    targetPoint.point = Point(*point[:3])
    transformed_point = tf_buffer.transform(targetPoint, target_frame, rospy.Duration(10.0))
    transformed_points[i] = vector_from_point(transformed_point.point, vertical=False)
  return transformed_points

def take_picture(file_path, bridge, rospy):
  image_msg = rospy.wait_for_message('/camera/image_color', Image)
  try:
    cv2_img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    cv2.imwrite(file_path, cv2_img)
  except CvBridgeError as e:
    print(e)

def save_seg_mask(file_path, faces, rospy, bridge):
  image_msg = rospy.wait_for_message('/camera/image_color', Image)
  try:
    cv2_img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    img = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
    for face in faces:
        cv2.fillPoly(img,[face],(255))
    cv2.imwrite(file_path, img)
  except Exception as e:
    print(e)

def publish_image_with_points(points, rospy, bridge, pub):
  image_msg = rospy.wait_for_message('/camera/image_color', Image)
  edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
  corners = points[1:]
  try:
    cv2_img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    cv2.circle(cv2_img, tuple(points[0]), 3, (255,0,0),4)

    for edge in edges_corners:
        start, end = corners[edge]
        cv2.line(cv2_img,tuple(start),tuple(end),(255,0,0),1)
        pub.publish(bridge.cv2_to_imgmsg(cv2_img, encoding='bgr8'))
  except CvBridgeError as e:
    print(e)

def publish_seg_mask(faces, rospy, bridge, pub):
  image_msg = rospy.wait_for_message('/camera/image_color', Image)
  try:
    cv2_img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    img = np.zeros(cv2_img.shape[:2], dtype=np.uint8)
    for face in faces:
        cv2.fillPoly(img,[face],(255))
    pub.publish(bridge.cv2_to_imgmsg(img, encoding='mono8'))
  except Exception as e:
    print(e)

def overlay_seg_mask(faces, rospy, bridge, pub):
  image_msg = rospy.wait_for_message('/camera/image_color', Image)
  try:
    cv2_img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    for face in faces:
        cv2.polylines(cv2_img,[face],True,(255,0,0),1)
    pub.publish(bridge.cv2_to_imgmsg(cv2_img, encoding='bgr8'))
  except Exception as e:
    print(e)

def proj_to_camera(points, rotation_matrix, translation_vector, camera_params):

    # camera parameters
    A = np.array(camera_params.K).reshape(3,3)
#    distortion_coeffs = np.array(camera_params.D)

    rt = np.concatenate((rotation_matrix, translation_vector), axis=1)

    camera_projection =(A.dot(rt)).dot(points)
    proj = np.empty((2, points.shape[1]), dtype='float32')
    proj[0, :] = camera_projection[0, :]/camera_projection[2, :]
    proj[1, :] = camera_projection[1, :]/camera_projection[2, :]

    return proj

def rotm_from_quaternion(quaternion):
    return tf.transformations.quaternion_matrix([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[:3,:3]

def vector_from_point(point, vertical=True):
    if vertical:
        return np.array([[point.x], [point.y], [point.z]])
    else:
        return np.array([point.x, point.y, point.z])

def rotate_random(m, low=-60, high=60.0):
    angle = low + np.random.sample()*(high-low)
    angle = angle*np.pi/180
    return m.dot([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])

def place_target_in_scene(target_centroid, target_size, rotation, scene, rospy, name='target'):
    target_pose = PoseStamped()
    target_pose.header.frame_id = 'world'
    target_pose.header.stamp = rospy.Time.now()
    target_pose.pose.position = target_centroid
    target_pose.pose.orientation =  Quaternion(*tf.transformations.quaternion_from_euler(0,0,rotation))

    scene.add_box(name, target_pose, size=target_size)
    return target_pose

def get_faces(mesh, rotation=0, offset=[0,0,0]):
    cos = np.cos(rotation)
    sin = np.sin(rotation)
    faces = []
    vertices = np.array(mesh.vertices)
    for indices in np.array(mesh.indices):
        corners = vertices[indices]
        corners = np.array([[x*cos - y*sin + offset[0], x*sin + y*cos + offset[1], z + offset[2]] for x,y,z in corners])
        corners = np.concatenate((np.transpose(corners), np.ones((1,len(indices)))), axis=0)
        faces.append(corners)
    
    return np.array(faces)

def get_3D_corners(vertices, rotation=0, offset=[0,0,0]):
    cos = np.cos(rotation)
    sin = np.sin(rotation)

    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = [[min_x, min_y, min_z],
               [min_x, min_y, max_z],
               [min_x, max_y, min_z],
               [min_x, max_y, max_z],
               [max_x, min_y, min_z],
               [max_x, min_y, max_z],
               [max_x, max_y, min_z],
               [max_x, max_y, max_z]]

    corners = np.array([[x*cos - y*sin + offset[0], x*sin + y*cos + offset[1], z + offset[2]] for x,y,z in corners])
    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def get_box_size(vertices):
    # width, length and height
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    return max_x - min_x, max_y - min_y, max_z - min_z

def add_robot_constraints():
    constraint = Constraints()
    constraint.name = "camera constraint"

    roll_constraint = OrientationConstraint()
    # 'base_link' is equal to the world link
    roll_constraint.header.frame_id = 'world'
    # The link that must be oriented upwards
    roll_constraint.link_name = "camera"
    roll_constraint.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0,np.pi/3,0))
    # Allow rotation of 45 degrees around the x and y axis
    roll_constraint.absolute_x_axis_tolerance = np.pi #Allow max rotation of x degrees
    roll_constraint.absolute_y_axis_tolerance = np.pi
    roll_constraint.absolute_z_axis_tolerance = np.pi/2
    # The roll constraint is the only constraint
    roll_constraint.weight = 1
    #constraint.orientation_constraints = [roll_constraint]

    joint_4_constraint = JointConstraint()
    joint_4_constraint.joint_name = "joint_4"
    joint_4_constraint.position = 0
    joint_4_constraint.tolerance_above = np.pi/2
    joint_4_constraint.tolerance_below = np.pi/2
    joint_4_constraint.weight = 1

    joint_5_constraint = JointConstraint()
    joint_5_constraint.joint_name = "joint_5"
    joint_5_constraint.position = np.pi/2
    joint_5_constraint.tolerance_above = np.pi/2
    joint_5_constraint.tolerance_below = np.pi/2
    joint_5_constraint.weight = 1

    joint_6_constraint = JointConstraint()
    joint_6_constraint.joint_name = "joint_6"
    joint_6_constraint.position = np.pi-0.79
    joint_6_constraint.tolerance_above = np.pi
    joint_6_constraint.tolerance_below = np.pi
    joint_6_constraint.weight = 1

    constraint.joint_constraints = [joint_4_constraint, joint_5_constraint, joint_6_constraint]
    return constraint

def generate_meshgrid(n_samples=5):
    # radii
    r = np.linspace(0.7, 0.9, n_samples)
    # rotation around y axis
    s = np.linspace(np.pi/4, np.pi/2.666, n_samples)
    # rotation around z axis
    t = np.linspace(-np.pi/4, np.pi/4, n_samples)
    rr, ss , tt = np.meshgrid(r,s,t)
    rr = rr.flatten()
    ss = ss.flatten()
    tt = tt.flatten()

    # by reversing half of this array, we prevent the robot
    # from going always right-to-left horizontally, which wastes
    # time since at the end of each row he has to go back all the
    # way to the right. Instead it goes left-to-right
 

    for j in range(0,len(tt),n_samples*2):
       tt[j:j+n_samples] = tt[j:j+n_samples][::-1]
    return rr, ss, tt

def point_difference(a,b):
    return np.linalg.norm(vector_from_point(a, vertical=False)-vector_from_point(b,vertical=False))
