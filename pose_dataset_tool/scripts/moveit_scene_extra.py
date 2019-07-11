#!/usr/bin/env python
import sys, rospy, moveit_commander, tf
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Quaternion

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('moveit_scene',anonymous=True)
scene = PlanningSceneInterface()
robot = moveit_commander.RobotCommander()
world_angle_offset = 0
rospy.sleep(2)
# a stand on which the target rests upon
box_pose = PoseStamped()
box_pose.header.frame_id = robot.get_planning_frame()
box_pose.header.stamp = rospy.Time.now()
box_pose.pose.position.x = 1
box_pose.pose.position.y = -0.085
box_pose.pose.position.z = 0.201
box_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0,0,world_angle_offset))
scene.add_box('box', box_pose, size=(0.18,0.45,0.402))

moveit_commander.roscpp_shutdown()
