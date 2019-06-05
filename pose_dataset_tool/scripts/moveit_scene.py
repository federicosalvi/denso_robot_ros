#!/usr/bin/env python
import sys, rospy, moveit_commander, tf
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Quaternion

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('moveit_scene',anonymous=True)
scene = PlanningSceneInterface()
robot = moveit_commander.RobotCommander()

rospy.sleep(2)
world_angle_offset = 0

plane_pose = PoseStamped()
plane_pose.header.frame_id = robot.get_planning_frame()
plane_pose.header.stamp = rospy.Time.now()
plane_pose.pose.position.z = -0.0151
plane_pose.pose.position.y = 0.45
plane_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0,0,world_angle_offset))
scene.add_box('plane', plane_pose, size=(4,2,0.03))

# this is just a piece of marble that we don't want the robot to smash into
marble_pose = PoseStamped()
marble_pose.header.frame_id = robot.get_planning_frame()
marble_pose.header.stamp = rospy.Time.now()
marble_pose.pose.position.x = 0
marble_pose.pose.position.y = 0.2875
marble_pose.pose.position.z = 0.065
marble_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0,0,world_angle_offset))
scene.add_box('marble', marble_pose, size=(4,0.295,0.14))

moveit_commander.roscpp_shutdown()
