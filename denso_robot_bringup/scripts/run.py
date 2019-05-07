#!/usr/bin/env python
import sys, rospy, tf, moveit_commander, random
from moveit_commander.planning_scene_interface import PlanningSceneInterface
#from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from moveit_msgs.msg import OrientationConstraint, JointConstraint, Constraints
import numpy as np

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

    R = np.vstack((forward, left, up)).T
    rospy.loginfo('\nRotation matrix:\n {}\n'.format(R))
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


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('vs_move_arm',anonymous=True)
scene = PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("arm")
robot = moveit_commander.RobotCommander()

rospy.sleep(2)

target = Point(1, 0, 0.114)

plane_pose = PoseStamped()
plane_pose.header.frame_id = robot.get_planning_frame()
#plane_pose.header.stamp = rospy.Time.now()
scene.add_plane('plane', plane_pose)

marble_pose = PoseStamped()
marble_pose.header.frame_id = robot.get_planning_frame()
#marble_pose.header.stamp = rospy.Time.now()
marble_pose.pose.position.x = 0
marble_pose.pose.position.y = 0.28575
marble_pose.pose.position.z = 0.0635
scene.add_box('marble', marble_pose, size=(4,0.2794,0.127))

driller_pose = PoseStamped()
driller_pose.header.frame_id = robot.get_planning_frame()
#driller_pose.header.stamp = rospy.Time.now()
driller_pose.pose.position = target
scene.add_box('driller', driller_pose, size=(0.226,0.0889,0.226))

group.set_max_acceleration_scaling_factor(0.1)
group.set_max_velocity_scaling_factor(0.25)


CAMERA_UPRIGHT = -0.79

group.set_joint_value_target({
  "joint_1": 0,
  "joint_2": 0,
  "joint_3": 0,
  "joint_4": 0,
  "joint_5": 0,
  "joint_6": CAMERA_UPRIGHT
})
# dummy initial pose
orient = Quaternion(*tf.transformations.quaternion_from_euler(0,0,0))
pose = Pose(Point( 0.75, 0, 0.9), orient)
#group.set_pose_target(pose)
success = group.go(True)
if not success:
  rospy.logfatal('Unable to go to starting position, exiting.')
  exit()
group.stop()

rospy.sleep(2)

constraint = Constraints()
constraint.name = "camera constraint"

roll_constraint = OrientationConstraint()
# 'base_link' is equal to the world link
roll_constraint.header.frame_id = robot.get_planning_frame()
# The link that must be oriented upwards
roll_constraint.link_name = "J6"
roll_constraint.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0,0,0))
# Allow rotation of 45 degrees around the x and y axis
roll_constraint.absolute_x_axis_tolerance = np.pi/4 #Allow max rotation of x degrees
roll_constraint.absolute_y_axis_tolerance = np.pi/4
roll_constraint.absolute_z_axis_tolerance = np.pi/4
# The roll constraint is the only constraint
roll_constraint.weight = 1
#constraint.orientation_constraints = [roll_constraint]

joint_5_constraint = JointConstraint()
joint_5_constraint.joint_name = "joint_5"
joint_5_constraint.position = 0
joint_5_constraint.tolerance_above = np.pi
joint_5_constraint.tolerance_below = np.pi/12
joint_5_constraint.weight = 1

joint_6_constraint = JointConstraint()
joint_6_constraint.joint_name = "joint_6"
joint_6_constraint.position = CAMERA_UPRIGHT
joint_6_constraint.tolerance_above = np.pi/2
joint_6_constraint.tolerance_below = np.pi/2
joint_6_constraint.weight = 1

constraint.joint_constraints = [joint_6_constraint] # [joint_5_constraint, joint_6_constraint]
group.set_path_constraints(constraint)

n_samples = 5
# radii
r = np.linspace(0.75, 0.9, n_samples)
# rotation around y axis
s = np.linspace(np.pi/4, np.pi/3, n_samples)
# rotation around z axis
t = np.linspace(-np.pi/4, np.pi/4, n_samples)
rr, ss , tt = np.meshgrid(r,s,t)
rr = rr.flatten()
ss = ss.flatten()
tt = tt.flatten()

i = 0
while not rospy.is_shutdown():
	pose.position = sample_sphere(rr[i], ss[i], tt[i])
        forward = [
		target.x - pose.position.x,
		target.y - pose.position.y,
		target.z - pose.position.z
	]
        forward = normalize(forward)

	pose.orientation = look_at(forward)

	group.set_pose_target(pose)
   	success = group.go(True)
        group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        group.clear_pose_targets()
	if not success:
	        rospy.logerr('\nFailed for position:\n\n{}\n\n'.format(pose))
	else:
		rospy.loginfo('\nMoved to position:\n{}\njoints values:\n{}\n'.format(pose,group.get_current_joint_values()))

	i += 1
        if i == len(rr):
		rospy.loginfo('Finished')
		break


moveit_commander.roscpp_shutdown()
