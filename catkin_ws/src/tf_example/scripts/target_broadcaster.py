#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
from std_msgs.msg import Time
import tf_conversions
import random

def send_tf_target():
    br = tf2_ros.TransformBroadcaster()
    pub = rospy.Publisher("/emulated_uav/target_found", Time, queue_size=10)

    rospy.sleep(2.0)  # Wait for system to start

    now = rospy.Time.now()

    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = now
    t.header.frame_id = "camera"
    t.child_frame_id = "target"
    t.transform.translation.x = random.uniform(0.2, 1.0)
    t.transform.translation.y = random.uniform(0.0, 0.5)
    t.transform.translation.z = random.uniform(0.0, 0.5)
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    br.sendTransform(t)
    pub.publish(Time(data=now))

    rospy.loginfo("Broadcasted target transform and published timestamp.")

if __name__ == '__main__':
    rospy.init_node('target_broadcaster')
    send_tf_target()
