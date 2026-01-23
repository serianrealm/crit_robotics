import yaml
import struct
from abc import abstractmethod, ABC
from functools import partial, lru_cache

from rclpy.logging import RcutilsLogger
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosidl_runtime_py import get_message

from .protocol import Protocol
from .rosidl_utils import (
    fill_message_from_values,
    flatten_message_to_values,
    message_type_to_struct_format,
    parse_url
)
from .udp_socket import UdpSocket

class UdpBridgeNodeInterface(Node, ABC):
    node_name = "udpsocket_python"
    def __init__(self):
        super().__init__(self.node_name)

        self.sock = UdpSocket(
            local_ip=self.get_parameter_or("local_ip", "127.0.0.1").value,
            local_port=self.get_parameter_or("local_port", 6006).value,
            remote_ip=self.get_parameter_or("remote_ip", "127.0.0.1").value,
            remote_port=self.get_parameter_or("remote_port", 6007).value,
            logger=self.logger
        )

        self._init_protocol()

        self.sock.on_message(self.uplink_callback)
        self.sock.start()

    @property
    def logger(self) -> RcutilsLogger:
        return self.get_logger()
    
    @abstractmethod
    def downlink_callback(self, msg, topic_name=None):
        raise NotImplementedError()
    
    @abstractmethod
    def uplink_callback(self, data, msg_id=None):
        raise NotImplementedError()

    def destroy_node(self):
        self.sock.stop()
        return super().destroy_node()

    @lru_cache
    def get_protocol(self, topic_name_or_msg_id: str|int) -> Protocol|None:
        if isinstance(topic_name_or_msg_id, int):
            for prot in self.PROTOCOL:
                if prot.msg_id == topic_name_or_msg_id:
                    return prot
            return None
        elif isinstance(topic_name_or_msg_id, str):
            for prot in self.PROTOCOL:
                if prot.topic_name == topic_name_or_msg_id:
                    return prot
            return None
        else:
            return None
    

    def _init_protocol(self):
        # magic, ver, seq, stamp, message_id
        self.magic_number = self.get_parameter_or("magic_number", 0xA55A).value
        self.version = self.get_parameter_or("version", 1).value
        self.sequence_number = 0

        url = parse_url(self.get_parameter_or(
            "url", f"package://{self.node_name}/config/channels.yaml")).value
        with open(url, "r", encoding="utf-8") as f:
            channels = yaml.safe_load(f)

        self.PROTOCOL : list[Protocol] = []
        for ch in channels:
            topic_name = ch["topic"]
            msg_id = ch["id"]
            msg_type = get_message(ch["type"])
            msg_direction = ch["direction"]
            msg_format = message_type_to_struct_format(msg_type, prefix="!") # big endian
            comm_entity = None

            if ch["direction"] == "uplink":
                comm_entity = self.create_publisher(
                    msg_type,
                    ch["topic"],
                    QoSProfile()
                )
            elif ch["direction"] == "downlink":
                comm_entity = self.create_subscription(
                    msg_type,
                    ch["topic"],
                    QoSProfile(),
                    partial(self.downlink_callback, topic_name=topic_name)
                )
            else:
                raise ValueError("key `direction` must be `uplink` or `downlink`")

            self.PROTOCOL.append(Protocol(
                msg_id=msg_id,
                topic_name=topic_name,
                msg_type=msg_type,
                msg_direction=msg_direction,
                msg_format=msg_format,
                comm_entity=comm_entity
            ))


class UdpBridge(UdpBridgeNodeInterface):
    def downlink_callback(self, msg, topic_name=None):
        protocol = self.get_protocol(topic_name)
        if protocol is None:
            self.logger.error(f"Unknown topic name: {topic_name}")
            return

        stamp = self.get_clock().now().nanoseconds

        header = struct.pack(
            self.get_parameter_or("header_format", "!HBIQH").value,
            self.magic_number,
            self.version,
            self.sequence_number,
            stamp,
            protocol.msg_id
        )
        
        payload = struct.pack(protocol.msg_format, *flatten_message_to_values(msg))
        data = header + payload

        self.sock.send(data)
        self.sequence_number += 1

    def uplink_callback(self, data, msg_id=None):
        # NOTE: !HBIQH -> big endian, uint16, uint8, uint32, uint64, uint16
        header_format = self.get_parameter_or("header_format", "!HBIQH").value
        header_size = struct.calcsize(header_format)

        if len(data) < header_size:
            self.logger.error("Got unexpected EOF while parsing datapack.")
            return
            
        magic_number, version, sequence_number, timestamp, message_id = struct.unpack_from(
            header_format,
            buffer=data,
            offset=0
        )

        if msg_id is not None:
            message_id = msg_id # override

        protocol = self.get_protocol(message_id)
        if protocol is None:
            self.logger.error(f"Unknown message id: {message_id}")
            return

        if version != self.version:
            self.logger.error("Version mismatch.")
            return

        if magic_number != self.magic_number:
            self.logger.error("Datapack transfer error: magic number mismatch.")
            return

        msg = fill_message_from_values(
            protocol.msg_type(),
            struct.unpack_from(
                protocol.msg_format,
                buffer=data,
                offset=header_size))
        
        publisher = protocol.comm_entity
        publisher.publish(msg)

        # TODO: check sequence number, calculate loss pack rate

        # TODO: check timestamp, calculate transfer delay

