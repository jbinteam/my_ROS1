#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
import tf
from scipy.spatial.transform import Rotation as R
from cv2 import aruco
from threading import Lock


def create_grid_board(marker_size, aruco_dict, init_id=0, nx=4, ny=4, gap=0.04):
    """Create a GridBoard and return its object, valid IDs, and dimensions."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)

    # OpenCV 4.2 / Noetic → use GridBoard_create
    board = cv2.aruco.GridBoard_create(
        markersX=nx,
        markersY=ny,
        markerLength=marker_size,
        markerSeparation=gap,
        dictionary=aruco_dict,
        firstMarker=int(init_id)
    )

    board_width = nx * marker_size + (nx - 1) * gap
    board_height = ny * marker_size + (ny - 1) * gap
    marker_ids = set(range(init_id, init_id + nx * ny))

    return board, marker_ids, board_width, board_height


def apply_tvec_offset(tvecs, rvecs, offset):
    """Apply an offset (in board frame) to translation vector."""
    Rm, _ = cv2.Rodrigues(rvecs)
    return tvecs + (Rm @ offset)


class MultiBoardTFNode:
    def __init__(self):
        rospy.init_node("multi_aruco_board_tf_node")

        # ---------------- Parameters ----------------
        image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_link")
        intrinsics_file = rospy.get_param("~intrinsics_file", "my_intrinsics.json")
        self.processing_rate = rospy.get_param("~processing_rate", 30.0)

        # ---------------- Load intrinsics ----------------
        if not os.path.exists(intrinsics_file):
            rospy.logerr(f"Camera intrinsics file not found: {intrinsics_file}")
            raise FileNotFoundError(intrinsics_file)

        with open(intrinsics_file, "r") as f:
            cam = json.load(f)

        self.camera_matrix = np.array([
            [float(cam["fx"]), 0.0, float(cam["cx"])],
            [0.0, float(cam["fy"]), float(cam["cy"])],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        if all(k in cam for k in ["k1", "k2", "p1", "p2", "k3"]):
            self.dist_coeffs = np.array(
                [cam[k] for k in ["k1", "k2", "p1", "p2", "k3"]],
                dtype=np.float32
            ).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1))

        # ---------------- Define boards ----------------
        self.boards = []
        for idx, start_id in enumerate([0, 16, 32], start=1):
            board, ids, w, h = create_grid_board(
                marker_size=0.04,
                aruco_dict=cv2.aruco.DICT_4X4_250,
                init_id=start_id
            )
            offset = np.array([w / 2, h / 2, 0.0]).reshape(3, 1)
            self.boards.append((f"board{idx}", board, ids, offset))

        # ---------------- Detector ----------------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.detector_params = cv2.aruco.DetectorParameters_create()

        # ---------------- ROS Interfaces ----------------
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("aruco/detected_image", Image, queue_size=1)

        self.tf_broadcaster = tf.TransformBroadcaster()

        # Shared frame buffer
        self.latest_frame = None
        self.frame_lock = Lock()
        self.zero_point = np.array([[0, 0, 0]], dtype=np.float32)

        rospy.loginfo(f"Multi-board ArUco detector initialized. Subscribing to {image_topic}")

        # Main loop
        self.loop()

    # ---------------- Callbacks ----------------
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.frame_lock:
            self.latest_frame = (frame, msg.header)

    def loop(self):
        rate = rospy.Rate(self.processing_rate)
        while not rospy.is_shutdown():
            self.process_frame()
            rate.sleep()

    def process_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame, header = self.latest_frame
            frame = frame.copy()

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.detector_params)
        if ids is None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))
            return

        ids = ids.flatten()

        for name, board, valid_ids, offset in self.boards:
            # Estimate pose of the board directly
            rvecs = np.zeros((3, 1))
            tvecs = np.zeros((3, 1))
            retval, rvecs, tvecs = cv2.aruco.estimatePoseBoard(
                corners, ids, board, self.camera_matrix, self.dist_coeffs, rvecs, tvecs
            )

            if retval <= 0:  # board not detected well enough
                continue

            # Apply offset (board center)
            new_tvec = apply_tvec_offset(tvecs, rvecs, offset)

            # Convert rotation to quaternion
            Rmat, _ = cv2.Rodrigues(rvecs)
            quat = R.from_matrix(Rmat).as_quat()  # [x,y,z,w]

            # Publish TF
            self.tf_broadcaster.sendTransform(
                (float(new_tvec[0]), float(new_tvec[1]), float(new_tvec[2])),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                rospy.Time.now(),
                name,
                self.camera_frame
            )

            # Draw axes
            cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs, tvecs, 0.05)
            cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs, new_tvec, 0.1)

            # Label board
            center_2d, _ = cv2.projectPoints(self.zero_point, rvecs, new_tvec, self.camera_matrix, self.dist_coeffs)
            center_2d = tuple(center_2d[0][0].astype(int))
            cv2.putText(frame, name, center_2d,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        annotated_msg.header = header
        self.image_pub.publish(annotated_msg)


if __name__ == "__main__":
    try:
        MultiBoardTFNode()
    except rospy.ROSInterruptException:
        pass
