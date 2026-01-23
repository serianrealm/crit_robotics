from dataclasses import dataclass

from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

@dataclass
class Protocol:
    msg_id: int
    topic_name: str
    msg_direction: str
    msg_format: str
    msg_type: type
    comm_entity: Publisher|Subscription