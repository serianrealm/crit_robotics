"""Common ROS 2 node interfaces for detector."""

from abc import abstractmethod, ABC

from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.logging import RcutilsLogger
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
import tf2_ros
from vision_msgs.msg import Detection2DArray

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

        self.tf_buffer = tf2_ros.Buffer(cache_time=1.0)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.stamp = None

        self.vision_raw_sub = self.create_subscription(
            Detection2DArray,
            "vision/raw",
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
            "vision/tracked",
            QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
        )

    def callback(self, msg: Detection2DArray):
        transform = self.tf_buffer.lookup_transform(
            target_frame="base_link",
            source_frame=msg.header.frame_id,
            time=msg.header.stamp
        ).transform.R

        raw_detections = []
        for det in msg.detections:
            det.pose.pose.orientation = Rotation.from_matrix(
                transform * Rotation.from_quat(det.pose.pose.orientation).as_matrix()).as_quat()
            raw_detections.append(TrackingObject(
                class_id=int(float(det.results[0].hypothesis.class_id)) % 10,
                score=float(det.results[0].hypothesis.score),
                bbox=cxcywh2xyxy(np.array([det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.size_x, det.bbox.size_y])),
                msg=det
            ))

        trackers = self.update(raw_detections)

        tracked_detections = []
        for trk in trackers:
            trk.msg.id = trk.id
            tracked_detections.append(trk.msg)

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
            conf_thres=0.5
        )

    def update(self, detections):
        return self.mot_tracker.update(detections)