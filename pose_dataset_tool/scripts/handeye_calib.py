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

tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
bridge = CvBridge()
rospy.sleep(2)


group.set_max_acceleration_scaling_factor(0.05)
group.set_max_velocity_scaling_factor(0.15)

# dummy initial pose
rotation = (np.pi/2,-np.pi/4,np.pi/2)
orient = Quaternion(*tf.transformations.quaternion_from_euler(*rotation))
pose = Pose(Point( 0.7, 0, 0.5), orient)
group.set_pose_target(pose)
success = group.go(True)
group.stop()

#group.set_path_constraints(add_robot_constraints())

dir = '/tmp/pictures/'
#for file in os.listdir(dir):
#  os.remove(dir + file)
n_samples=5
x = np.linspace(0.35, 0.6, n_samples)
# rotation around y axis
y = np.linspace(-.4, 0, n_samples)
# rotation around z axis
z = np.linspace(0.2, 0.4, n_samples)
xx, yy, zz = np.meshgrid(x,y,z)

xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()

def add_random_rotation(roll, pitch, yaw):
   roll += np.random.random()*0.25
   pitch += -0.1 + np.random.random()*0.2
   yaw += -0.1 + np.random.random()*0.2
   return roll, pitch, yaw
i = 0
j = 0
rts = []
while not rospy.is_shutdown():
        pose.position = Point(xx[i],yy[i],zz[i]) # , offset=sphere_origin)
        pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(*add_random_rotation(*rotation)))

        group.set_pose_target(pose)
        success = group.go(True, wait=True)
        group.stop()
        group.clear_pose_targets()

        rospy.sleep(2)

        if not success:
            actual = group.get_current_pose().pose.position
            difference = point_difference(pose.position,actual)
            if difference > 1e-02:
                rospy.logerr('\nFailed for position:\n{}\n'.format(pose))
		i+=1
                continue

	transform = tf_buffer.lookup_transform_full('base_link', rospy.Time(), 'J6', rospy.Time(), 'world', rospy.Duration(10.0))
	t = vector_from_point(transform.transform.translation, vertical=False)
	r = rotm_from_quaternion(transform.transform.rotation)
	rts.append((r,t))
        rospy.sleep(2)

        take_picture('{}image{}.png'.format(dir,j), bridge, rospy)
	j += 1
        i += 1
	if i == len(xx):
		break

f = open('{}robot_cali.txt'.format(dir),'w+')
output = '{}\n'.format(j)
for rt in rts:
	r = rt[0]
	t = rt[1]
	for i in range(3):
		for j in range(3):
			output += '{} '.format(r[i,j])
		output += '{}\n'.format(t[i]*1000)
	output += '0.000000 0.000000 0.000000 1.000000\n\n'

f.write(output)
f.close()
f = open('{}robot_cali_t.txt'.format(dir),'w+')
output = '{}\n'.format(j)
for rt in rts:
	r = rt[0].T
	t = rt[1]
	for i in range(3):
		for j in range(3):
			output += '{} '.format(r[i,j])
		output += '{}\n'.format(t[i]*1000)
	output += '0.000000 0.000000 0.000000 1.000000\n\n'

f.write(output)
f.close()
moveit_commander.roscpp_shutdown()
