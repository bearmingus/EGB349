import rospy
import cv2
import depthai as dai
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

# Setup for DepthAI camera
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# ROS node setup
rospy.init_node('aruco_detector_node')
pub = rospy.Publisher('/processed/image_raw', Image, queue_size=10)
bridge = CvBridge()

# Start the camera
with dai.Device(pipeline) as device:
    video = device.getOutputQueue("video", 4, False)

    while not rospy.is_shutdown():
        frame = video.get().getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Draw detected markers
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Convert frame to ROS message and publish
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(msg)
        
        # Optional: display the image locally on Pi
        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

