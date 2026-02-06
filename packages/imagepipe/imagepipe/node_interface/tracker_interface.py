"""Common ROS 2 node interfaces for detector."""

from copy import deepcopy
from abc import abstractmethod, ABC

from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.time import Duration
from rclpy.logging import RcutilsLogger
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
import tf2_ros
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Point, Quaternion

from ..solutions.bytetrack import TrackingObject, ByteTrack
import numpy as np

def cxcywh2xyxy(bboxes: np.ndarray):
    """Convert bounding boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    
    Args:
        bboxes: Array of shape [batch_size, 4] with format [cx, cy, w, h]
    
    Returns:
        Array of shape [batch_size, 4] with format [x1, y1, x2, y2]
    """
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


class TrackerNodeInterface(Node, ABC):
    """
    Docstring for TrackerNodeInterface
    """
    node_name = "tracker"

    def __init__(self):
        super().__init__(
            node_name=self.node_name,
            automatically_declare_parameters_from_overrides=True
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=1.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.stamp = None

        self.vision_raw_sub = self.create_subscription(
            Detection2DArray,
            "/vision/raw",
            self.callback,
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
        )

        self.vision_tracked_pub = self.create_publisher(
            Detection2DArray,
            "/vision/tracked",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
        )

    def callback(self, msg: Detection2DArray):
        header = Header(frame_id="base_link", stamp=msg.header.stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame=header.frame_id,
                source_frame=msg.header.frame_id,
                time=msg.header.stamp,
            ).transform
        except Exception as ex:
            self.logger.warning(f"topic `/tf` does not appear to be published correctly yet: {ex}", once=True)
            return
        
        translation = np.array([
            transform.translation.x, 
            transform.translation.y, 
            transform.translation.z
        ])

        rotation = Rotation.from_quat([
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        ]).as_matrix()

        raw_detections = []
        self.logger.error(f"num dets: {len(msg.detections)}")
        for det in msg.detections:
            det.header = header
            pose = det.results[0].pose.pose

            position = translation + rotation @ np.array([
                pose.position.x, pose.position.y, pose.position.z])

            orientation = rotation @ Rotation.from_quat([pose.orientation.x, 
                pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
            
            quaternion = Rotation.from_matrix(orientation).as_quat()

            # for debug only
            # normal_vector = orientation @ (np.array([0, 0, 1]).reshape(3, 1))
            # normal_vector = normal_vector.reshape(-1)

            # self.logger.error(f"")
            # self.logger.error(f"normal vector: {normal_vector}")
            # self.logger.error(f"dis: {position}")

            # pitch = -np.atan(normal_vector[2]/np.hypot(normal_vector[0], normal_vector[1]))
            # yaw = np.atan(normal_vector[1]/normal_vector[0])

            # normal_vector2 = Rotation.from_quat([pose.orientation.x, 
            #     pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix() @ (np.array([0, 0, 1]).reshape(3, 1))
            # normal_vector2 = normal_vector2.reshape(-1)

            # yaw2 = np.atan(normal_vector2[0]/normal_vector2[2])
            # self.logger.error(f"error : {normal_vector}")

            # self.logger.error(f"normal vector: {normal_vector}")
            # self.logger.error(f"pitch yaw: {pitch}, {yaw}")

            det.header = msg.header
            pose.position = Point(
                x=position[0],
                y=position[1],
                z=position[2]
            )

            pose.orientation = Quaternion(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3]
            )

            # pose.orientation = Quaternion(
            #     x=pitch,
            #     y=yaw,
            #     z=yaw2,
            #     w=0.
            # )
            
            raw_detections.append(TrackingObject(
                class_id=int(float(det.results[0].hypothesis.class_id)) % 10,
                score=float(det.results[0].hypothesis.score),
                bbox=cxcywh2xyxy(np.array([
                    det.bbox.center.position.x, 
                    det.bbox.center.position.y,
                    det.bbox.size_x, 
                    det.bbox.size_y])),
                message=det
            ))

        trackers = self.update(raw_detections)
        tracked_detections = []
        for trk in trackers:
            m = trk.message
            m.id = str(int(trk.id))
            tracked_detections.append(m)

        # self.logger.info(f"{tracked_detections}")

        self.vision_tracked_pub.publish(Detection2DArray(
            header=msg.header,
            detections=tracked_detections
        ))

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError()

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()


class MotTracker(TrackerNodeInterface):
    """"""

    def __init__(self):
        super().__init__()  
        
        self.mot_tracker = ByteTrack(
            min_hits=3,
            iou_thres=0.25,
            conf_thres=0.7
        )

    def update(self, detections):
        return self.mot_tracker.update(detections)