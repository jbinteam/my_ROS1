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
from std_msgs.msg import Float32
from geometry_msgs.msg import Point


def create_grid_board(marker_size, aruco_dict, init_id=0, nx=4, ny=4, gap=0.04):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
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
    Rm, _ = cv2.Rodrigues(rvecs)
    return tvecs + (Rm @ offset)


class MultiBoardTFNode:
    def __init__(self):
        rospy.init_node("multi_aruco_board_tf_node")

        image_topic = rospy.get_param("~image_topic", "/k4a/rgb/image_raw")
        self.camera_frame = rospy.get_param("~camera_frame", "rgb_camera_link")
        intrinsics_file = rospy.get_param("~intrinsics_file", "my_intrinsics.json")
        self.processing_rate = rospy.get_param("~processing_rate", 30.0)

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

        # Define 2 boards
        self.boards = []
        for idx, start_id in enumerate([0, 16], start=1):
            board, ids, w, h = create_grid_board(
                marker_size=0.04,
                aruco_dict=cv2.aruco.DICT_4X4_250,
                init_id=start_id
            )
            offset = np.array([w / 2, h / 2, 0.0]).reshape(3, 1)
            self.boards.append((f"board{idx}", board, ids, offset))

        # Detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.detector_params = cv2.aruco.DetectorParameters_create()

        # ROS Interfaces
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("aruco/detected_image", Image, queue_size=1)

        self.tf_broadcaster = tf.TransformBroadcaster()

        # New publishers
        self.angle_pub = rospy.Publisher("aruco/board1_board2/z_angle", Float32, queue_size=1)
        self.board2_tvec_pub = rospy.Publisher("aruco/board2/new_tvec", Point, queue_size=1)

        self.latest_frame = None
        self.frame_lock = Lock()
        self.zero_point = np.array([[0, 0, 0]], dtype=np.float32)

        rospy.loginfo(f"Multi-board ArUco detector initialized. Subscribing to {image_topic}")
        self.loop()

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

        # corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.detector_params)
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict)
        if ids is None:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))
            return

        ids = ids.flatten()
        board_poses = {}

        # Estimate pose of each board
        for name, board, valid_ids, offset in self.boards:
            rvecs = np.zeros((3, 1))
            tvecs = np.zeros((3, 1))
            retval, rvecs, tvecs = cv2.aruco.estimatePoseBoard(
                corners, ids, board, self.camera_matrix, self.dist_coeffs, rvecs, tvecs
            )
            if retval <= 0:
                continue

            new_tvec = apply_tvec_offset(tvecs, rvecs, offset)
            Rmat, _ = cv2.Rodrigues(rvecs)
            board_poses[name] = (Rmat, new_tvec)

            quat = R.from_matrix(Rmat).as_quat()
            self.tf_broadcaster.sendTransform(
                (float(new_tvec[0]), float(new_tvec[1]), float(new_tvec[2])),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                rospy.Time.now(),
                name,
                self.camera_frame
            )

            cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs, tvecs, 0.1)
            cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs, new_tvec, 0.3)

        # Compute angle between board1 and board2
        if "board1" in board_poses and "board2" in board_poses:
            R1, _ = board_poses["board1"]
            R2, new_tvec2 = board_poses["board2"]

            z1 = R1[:, 2]
            z2 = R2[:, 2]
            cos_angle = np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            # Publish angle
            self.angle_pub.publish(Float32(data=float(angle_deg)))

            # Publish board2 translation
            self.board2_tvec_pub.publish(Point(
                x=float(new_tvec2[0]),
                y=float(new_tvec2[1]),
                z=float(new_tvec2[2])
            ))

            # Overlay on image
            cv2.putText(frame, f"Angle Z(board1,board2): {angle_deg:.2f} deg",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        annotated_msg.header = header
        self.image_pub.publish(annotated_msg)


if __name__ == "__main__":
    try:
        MultiBoardTFNode()
    except rospy.ROSInterruptException:
        pass
