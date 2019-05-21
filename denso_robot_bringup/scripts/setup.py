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

def normalize(v):
    v = np.array(v)
    return v/np.linalg.norm(v)

def sample_sphere(r, s, t):
    x = r * np.cos(t) * np.sin(s)
    y = r * np.sin(t) * np.sin(s)
    z = r * np.cos(s)
    return Point(x,y,z)

def look_at(forward):
    camera_up = np.array([0,0,1])
    left = normalize(np.cross(camera_up, forward))
    up = np.cross(forward, left)

    # we're switching the axis since the camera "points" along the z-axis
    R = np.vstack((-left, -up, forward)).T

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

def transform_pose(pose, target_frame='camera', source_frame='world'):
  targetPose = PoseStamped()
  targetPose.header.stamp = rospy.Time.now()
  targetPose.header.frame_id = source_frame
  targetPose.pose = pose
  rospy.sleep(2)

  transform = tf_buffer.lookup_transform_full(target_frame, rospy.Time(), source_frame, rospy.Time(), 'world', rospy.Duration(10.0))
  transformed_pose = tf2_geometry_msgs.do_transform_pose(targetPose, transform).pose
  # wait until the camera is in position
  while np.abs(transformed_pose.position.y) > 0.001 and not rospy.is_shutdown():
    transform = tf_buffer.lookup_transform_full(target_frame, rospy.Time().now(), source_frame, rospy.Time(), 'world', rospy.Duration(10.0))
    transformed_pose = tf2_geometry_msgs.do_transform_pose(targetPose, transform).pose
  return transform, transformed_pose

def transform_points(points, target_frame='camera', source_frame='world'):
  transformed_points = np.empty((3, points.shape[1]))
  for i,point in enumerate(points):
    targetPoint = PointStamped()
    targetPoint.header.stamp = rospy.Time.now()
    targetPoint.header.frame_id = source_frame
    targetPoint.point = Point(*point[:3])
    transformed_point = tf_buffer.transform(targetPoint, target_frame, rospy.Duration(10.0))
    transformed_points[i] = vector_from_point(transformed_point, vertical=False)
  return transformed_points

def take_picture(file_path):
  image_msg = rospy.wait_for_message('/camera/image_color', Image)
  try:
    cv2_img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    cv2.imwrite(file_path, cv2_img)
  except CvBridgeError as e:
    print(e)

pub = rospy.Publisher('/camera/camera_with_points', Image, queue_size=10)

def publish_image_with_points(points):
  image_msg = rospy.wait_for_message('/camera/image_color', Image)
  edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
  corners = points[1:]
  try:
    cv2_img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    cv2.circle(cv2_img, tuple(points[0]), 1, (255,0,0),4)
    cv2.line(cv2_img, (0, cv2_img.shape[0]/2), (cv2_img.shape[1], cv2_img.shape[0]/2), (0,0,255),2)
    cv2.line(cv2_img, (cv2_img.shape[1]/2, 0), (cv2_img.shape[1]/2, cv2_img.shape[0]), (0,0,255),2)

    for edge in edges_corners:
	start, end = corners[edge]
	cv2.line(cv2_img,tuple(start),tuple(end),(255,0,0),3)
    pub.publish(bridge.cv2_to_imgmsg(cv2_img, encoding='bgr8'))
  except CvBridgeError as e:
    print(e)

def proj_to_camera(points, rotation_matrix, translation_vector, camera_params):

	# camera parameters
	A = np.array(camera_params.K).reshape(3,3)
#	distortion_coeffs = np.array(camera_params.D)

	rt = np.concatenate((rotation_matrix, translation_vector), axis=1)
	rospy.loginfo('rt:\n{}'.format(rt))

	camera_projection =(A.dot(rt)).dot(points)
	proj = np.empty((2, points.shape[1]), dtype='float32')
	proj[0, :] = camera_projection[0, :]/camera_projection[2, :]
        proj[1, :] = camera_projection[1, :]/camera_projection[2, :]

	return proj

def load_camera_parameters():
	return rospy.wait_for_message("/camera/camera_info", CameraInfo)

def rotm_from_quaternion(quaternion):
	return tf.transformations.quaternion_matrix([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[:3,:3]

def vector_from_point(point, vertical=True):
	if vertical:
		return np.array([[point.x], [point.y], [point.z]])
	else:
		return np.array([point.x, point.y, point.z])

def rotate_random(m, low=-60, high=60.0):
    angle = low + np.random.sample()*(high-low)
    rospy.loginfo('\nangle of rotation:{}\n'.format(angle))
    angle = angle*np.pi/180
    return m.dot([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])

camera = image_geometry.PinholeCameraModel()

def project(point):
    targetPoint = PointStamped()
    targetPoint.header.stamp = rospy.Time.now()
    targetPoint.header.frame_id = 'world'
    targetPoint.point = Point(*point[:3])
    transformed_point = tf_buffer.transform(targetPoint, 'camera', rospy.Duration(10.0))

    return camera.rectifyPoint(camera.project3dToPixel(vector_from_point(transformed_point.point)))

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('vs_move_arm',anonymous=True)
scene = PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("arm")
robot = moveit_commander.RobotCommander()

camera_params = load_camera_parameters()
camera.fromCameraInfo(camera_params)
tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
bridge = CvBridge()

rospy.sleep(2)

plane_pose = PoseStamped()
plane_pose.header.frame_id = robot.get_planning_frame()
plane_pose.header.stamp = rospy.Time.now()
scene.add_plane('plane', plane_pose)

# this is just a piece of marble that we don't want the robot to smash into
marble_pose = PoseStamped()
marble_pose.header.frame_id = robot.get_planning_frame()
marble_pose.header.stamp = rospy.Time.now()
marble_pose.pose.position.x = 0
marble_pose.pose.position.y = 0.28575
marble_pose.pose.position.z = 0.0635
scene.add_box('marble', marble_pose, size=(4,0.2794,0.127))


def get_3D_corners(vertices, origin=[0,0,0]):

    min_x = np.min(vertices[0,:]) + origin[0]
    max_x = np.max(vertices[0,:]) + origin[0]
    min_y = np.min(vertices[1,:]) + origin[1]
    max_y = np.max(vertices[1,:]) + origin[1]
    min_z = np.min(vertices[2,:]) + origin[2]
    max_z = np.max(vertices[2,:]) + origin[2]

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def get_box_size(corners):
    # width, length and height
    return corners[0,-1] - corners[0,0], corners[1,-1] - corners[1,0], corners[2,-1] - corners[2,0]

mesh = MeshPly('/root/catkin_ws/bracket.ply')
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
target_corners = get_3D_corners(vertices, [1,0,0])
target_width, target_length, target_height = get_box_size(target_corners)
rospy.loginfo(get_box_size(target_corners))
target_centroid = Point(1, 0, target_height/2)

target_points = np.hstack((np.vstack((vector_from_point(target_centroid),[1])),target_corners))
rospy.loginfo(target_points)
target_pose = PoseStamped()
target_pose.header.frame_id = robot.get_planning_frame()
target_pose.header.stamp = rospy.Time.now()
target_pose.pose.position = target_centroid
target_pose.pose.orientation =  Quaternion(*tf.transformations.quaternion_from_euler(0,0,0))
#half_target_width = 0.113
#half_target_length = 0.0444
#half_target_height = 0.113

#target_corners = np.array([
#	[target_centroid.x - half_target_width, target_centroid.y - half_target_length, target_centroid.z - half_target_height], # min x, min y, min z
#	[target_centroid.x - half_target_width, target_centroid.y - half_target_length, target_centroid.z + half_target_height], # min x, min y, max z
#	[target_centroid.x - half_target_width, target_centroid.y + half_target_length, target_centroid.z - half_target_height], # min x, max y, min z
#	[target_centroid.x - half_target_width, target_centroid.y + half_target_length, target_centroid.z + half_target_height], # min x, max y, max z
#	[target_centroid.x + half_target_width, target_centroid.y - half_target_length, target_centroid.z - half_target_height], # max x, min y, min z
#	[target_centroid.x + half_target_width, target_centroid.y - half_target_length, target_centroid.z + half_target_height], # max x, min y, max z
#	[target_centroid.x + half_target_width, target_centroid.y + half_target_length, target_centroid.z - half_target_height], # max x, max y, min z
#	[target_centroid.x + half_target_width, target_centroid.y + half_target_length, target_centroid.z + half_target_height]  # max x, max y, max z
#])

scene.add_box('target', target_pose, size=(target_width, target_length, target_height))

group.set_max_acceleration_scaling_factor(0.1)
group.set_max_velocity_scaling_factor(0.25)

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

constraint = Constraints()
constraint.name = "camera constraint"

roll_constraint = OrientationConstraint()
# 'base_link' is equal to the world link
roll_constraint.header.frame_id = robot.get_planning_frame()
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
joint_6_constraint.position = CAMERA_UPRIGHT
joint_6_constraint.tolerance_above = np.pi
joint_6_constraint.tolerance_below = np.pi
joint_6_constraint.weight = 1

constraint.joint_constraints = [joint_4_constraint, joint_5_constraint, joint_6_constraint]
group.set_path_constraints(constraint)

n_samples = 5
# radii
r = np.linspace(0.9, 0.9, n_samples)
# rotation around y axis
s = np.linspace(np.pi/4, np.pi/3, n_samples)
# rotation around z axis
t = np.linspace(-np.pi/4, np.pi/4, n_samples)
rr, ss , tt = np.meshgrid(r,s,t)
rr = rr.flatten()
ss = ss.flatten()
tt = tt.flatten()

dir = '/tmp/pictures/'
for file in os.listdir(dir):
  os.remove(dir + file)

i = 0
while not rospy.is_shutdown():
	pose.position = sample_sphere(rr[i], ss[i], tt[i])
        forward = normalize([
		target_centroid.x - pose.position.x,
		target_centroid.y - pose.position.y,
		target_centroid.z - pose.position.z
	])
	pose.orientation = look_at(forward)

	group.set_pose_target(pose)
   	success = group.go(True, wait=True)
        group.stop()
        group.clear_pose_targets()
	rospy.sleep(2)

	#rospy.loginfo('\nMoved to position:\n{}\njoints values:\n{}\n'.format(pose,group.get_current_joint_values()))

	frame_transform, object_pose = transform_pose(target_pose.pose)
	rospy.loginfo('\nFrame transform:\n{}\n'.format(frame_transform))
	rospy.loginfo('\nObject center transform:\n{}\n'.format(object_pose))
	rotation_matrix = rotm_from_quaternion(frame_transform.transform.rotation)
	translation_vector = vector_from_point(frame_transform.transform.translation)
	rospy.loginfo('\nTransform:\n\trotation matrix:\n{}\n\ttranslation vector:\n{}\n'.format(rotation_matrix, translation_vector))

	# project 3d points onto image plane
	#transformed_target_corners = transform_points(target_corners)
	proj_points = proj_to_camera(target_points, rotation_matrix, translation_vector, camera_params).T
	rospy.loginfo('Projected points:\n{}'.format(proj_points))
        proj_points = []
        for point in target_points.T:
                proj_point = project(point)
                proj_points.append([int(proj_point[0]), int(proj_point[1])])

	publish_image_with_points(np.array(proj_points))
	i += 1
	if i == len(rr):
		i = 0

moveit_commander.roscpp_shutdown()
