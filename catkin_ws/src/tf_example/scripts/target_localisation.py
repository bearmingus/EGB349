#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
from std_msgs.msg import Time

def callback(msg_in):
    try:
        t = tfBuffer.lookup_transform("map", "target", msg_in.data, rospy.Duration(0.5))
        rospy.loginfo("Target position in map frame: x=%.2f, y=%.2f, z=%.2f",
                      t.transform.translation.x,
                      t.transform.translation.y,
                      t.transform.translation.z)
    except Exception as e:
        rospy.logwarn("Could not find transform: %s", str(e))

if __name__ == '__main__':
    rospy.init_node('target_localizer')
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.Subscriber("/emulated_uav/target_found", Time, callback)
    rospy.loginfo("target_localizer running. Waiting for timestamp...")
    rospy.spin()
