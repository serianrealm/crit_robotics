"""Common ROS 2 node interfaces for detector."""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from abc import abstractmethod, ABC

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.node import Node
from rclpy.logging import RcutilsLogger
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from ament_index_python import get_package_share_directory
from sensor_msgs.msg import (
    Image,
    CameraInfo
)
from geometry_msgs.msg import (
    Point,
    Quaternion,
    Pose,
    PoseWithCovariance
)
from vision_msgs.msg import (
    Pose2D,
    Point2D,
    BoundingBox2D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    Detection2D,
    Detection2DArray
)

from ..runtime.pipelines import Yolov10PosePipeline


GROUNDING_SMALL_ARMOR = [
    (-0.0675, -0.027,  0.0036),
    (-0.0675,  0.027, -0.0036),
    ( 0.0675,  0.027, -0.0036),
    ( 0.0675, -0.027,  0.0036),
]

GROUNDING_BIG_ARMOR = [
    (-0.115, -0.029, -0.0038),
    (-0.115,  0.029,  0.0038),
    ( 0.115,  0.029,  0.0038),
    ( 0.115, -0.029, -0.0038),
]

ENERGY_ARMOR = [
    ( 0.00, -0.85, 0.),
    (-0.15, -0.70, 0.),
    ( 0.00, -0.55, 0.),
    ( 0.15, -0.70, 0.),
]

DRONE_ARMOR = [
    (-0.025, -0.05, 0.),
    (-0.025,  0.05, 0.),
    ( 0.025,  0.05, 0.),
    ( 0.025, -0.05, 0.),
]

OUTPOST_ARMOR = [
    (-0.0675, -0.027, -0.0035),
    (-0.0675,  0.027,  0.0035),
    ( 0.0675,  0.027,  0.0035),
    ( 0.0675, -0.027, -0.0035),
]

BASE_ARMOR = [
    (-0.0675, -0.028, 0.),
    (-0.0675,  0.028, 0.),
    ( 0.0675,  0.028, 0.),
    ( 0.0675, -0.028, 0.),
]

def camera_image_to_cv2(camera_image: Image) -> np.ndarray:
    encoding = (camera_image.encoding or "").lower()
    height, width, step = int(camera_image.height), int(camera_image.width), int(camera_image.step)

    if encoding in ("bgr8", "rgb8"):
        channels = 3
        image = np.ndarray(
            shape=(height, width, channels),
            dtype=np.uint8,
            buffer=memoryview(camera_image.data),
            strides=(step, channels, 1),
        )
        if encoding == "rgb8":
            image = image[:, :, ::-1].copy()
        return image

    if encoding in ("bgra8", "rgba8"):
        channels = 4
        image = np.ndarray(
            shape=(height, width, channels),
            dtype=np.uint8,
            buffer=memoryview(camera_image.data),
            strides=(step, channels, 1),
        )
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR,
        )

    if encoding == "mono8":
        image = np.ndarray(
            shape=(height, width),
            dtype=np.uint8,
            buffer=memoryview(camera_image.data),
            strides=(step, 1),
        )
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding: {camera_image.encoding!r}")

class DetectorNodeInterface(Node, ABC):
    """Base node that declares parameters up-front and exposes helpers."""
    node_name = "imagepipe"

    def __init__(self):
        super().__init__(
            node_name=self.node_name,
            automatically_declare_parameters_from_overrides=True
        )

        self.image_subs = [
            self.create_subscription(
                Image, 
                f"{topic_prefix}/image_raw", 
                partial(self.callback, topic_name=topic_prefix), 
                10
            ) for topic_prefix in self.get_parameter("subscriptions").value
        ]

        self.camera_info_subs = [
            self.create_subscription(
                CameraInfo, 
                f"{topic_prefix}/camera_info", 
                partial(self.camera_info_callback, topic_name=topic_prefix), 
                10
            ) for topic_prefix in self.get_parameter("subscriptions").value
        ]

        self.camera_infos = {
            topic_prefix: None 
            for topic_prefix in  self.get_parameter("subscriptions").value
        }

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()

    @abstractmethod
    def callback(*args, **kwargs):
        raise NotImplementedError()
    
    def get_camera_info(self, topic_name:str):
        return self.camera_infos.get(topic_name, None)

    def camera_info_callback(self, camera_info:CameraInfo, topic_name:str|None=None):
        if topic_name is None:
            raise ValueError("Invalid topic name.")
        self.camera_infos[topic_name] = camera_info

class YoloPoseDetector(DetectorNodeInterface):
    """"""

    def __init__(self):
        super().__init__()
        
        model_path = os.path.join(
            get_package_share_directory("imagepipe"),
            self.get_parameter("model_name_or_path").value
        )

        self.model = Yolov10PosePipeline(
            model_path,
            image_size=(640, 640)
        )

        self.vision_raw_pub = self.create_publisher(
            Detection2DArray,
            "vision/camera_optical_frame",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
        )

    def callback(self, camera_image:Image, topic_name:str|None=None):
        camera_info = self.get_camera_info(topic_name)
        if camera_info is None:
            self.logger.warning(f"Waiting for camera info {topic_name}/camera_info to synchronize", throttle_duration_sec=1.0, skip_first=True)
            return

        img = camera_image_to_cv2(camera_image)

        prediction = self.model(img)

        if len(prediction.shape) == 3:
            prediction = prediction.reshape(-1, *prediction.shape[2:])

        camera_matrix = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
        distortion_coefficients = np.array(camera_info.d, dtype=np.float64)

        msg = Detection2DArray(header=camera_image.header)

        for pred in prediction:
            image_points = pred[6:14].reshape(-1, 2)
            object_points = GROUNDING_SMALL_ARMOR

            match int(pred[5]) % 10:
                case 1:
                    object_points = GROUNDING_BIG_ARMOR
                case 5:
                    object_points = ENERGY_ARMOR
                case 6:
                    object_points = DRONE_ARMOR
                case 7:
                    object_points = OUTPOST_ARMOR
                case 8:
                    object_points = BASE_ARMOR
                case _:
                    object_points = GROUNDING_SMALL_ARMOR
            
            
            is_success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE
            )
            
            if not is_success:
                self.logger.error(
                    f"cv2.solvePnP failed for object_points={object_points}, \
                    image_points={image_points}, \
                    camera_matrix={camera_matrix.tolist()}, \
                    dist_coeffs={distortion_coefficients.tolist()}.",
                    throttle_duration_sec=3.0
                )
                continue
            else:
                point = tvec.reshape(-1)
                position = Point(
                    x=point[0],
                    y=point[1],
                    z=point[2]
                )

                quaternion = Rotation.from_rotvec(rvec.reshape(-1)).as_quat()
                orientation=Quaternion(
                    x=quaternion[0],
                    y=quaternion[1],
                    z=quaternion[2],
                    w=quaternion[3]
                )

                msg.detections.append(Detection2D(
                    header=camera_image.header,
                    results=[ObjectHypothesisWithPose(
                        hypothesis=ObjectHypothesis(
                            class_id=str(int(pred[5])),
                            score=float(pred[4])
                        ),
                        pose=PoseWithCovariance(
                            pose=Pose(
                                position=position,
                                orientation=orientation
                            )
                        )
                    )],
                    bbox=BoundingBox2D(
                        center=Pose2D(
                            position=Point2D(
                                x=pred[0],
                                y=pred[1]
                            )
                        ),
                        size_x=pred[2],
                        size_y=pred[3]
                    )
                ))

        self.vision_raw_pub.publish(msg)