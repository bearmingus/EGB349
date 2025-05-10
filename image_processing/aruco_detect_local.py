import cv2
import depthai as dai
import numpy as np

# Load predefined dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

# OAK-D camera setup
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam_rgb.preview.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    video = device.getOutputQueue("video", 4, False)

    while True:
        in_frame = video.get()
        frame = in_frame.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                print(f"ID: {ids[i][0]} | Center: ({center_x}, {center_y})")

        # cv2.imshow("ArUco Marker Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

cv2.destroyAllWindows()
