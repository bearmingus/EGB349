#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import tf_conversions

def broadcast_transforms():
    rospy.init_node('tf2_broadcaster_frames')
    br = tf2_ros.TransformBroadcaster()
    static_br = tf2_ros.StaticTransformBroadcaster()

    # UAV position in map frame
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "map"
    t.child_frame_id = "uav"
    t.transform.translation.x = 0.0
    t.transform.translation.y = 0.0
    t.transform.translation.z = 1.0
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    # UAV → camera transform (static)
    static_t = geometry_msgs.msg.TransformStamped()
    static_t.header.stamp = rospy.Time.now()
    static_t.header.frame_id = "uav"
    static_t.child_frame_id = "camera"
    static_t.transform.translation.x = 0.1
    static_t.transform.translation.y = 0.0
    static_t.transform.translation.z = -0.1
    q = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
    static_t.transform.rotation.x = q[0]
    static_t.transform.rotation.y = q[1]
    static_t.transform.rotation.z = q[2]
    static_t.transform.rotation.w = q[3]

    static_br.sendTransform(static_t)
    rospy.loginfo("tf2_broadcaster_frames running.")

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        t.header.stamp = rospy.Time.now()
        br.sendTransform(t)
        rate.sleep()

if __name__ == '__main__':
    broadcast_transforms()
