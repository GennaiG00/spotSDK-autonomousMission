import argparse
import logging
import os
import sys
import time

import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.api.mission
import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry as geometry
import bosdyn.mission.client
import bosdyn.util
from bosdyn.api import geometry_pb2, image_pb2, world_object_pb2
from bosdyn.api.autowalk import walks_pb2
from bosdyn.api.data_acquisition_pb2 import AcquireDataRequest, DataCapture, ImageSourceCapture
from bosdyn.api.graph_nav import graph_nav_pb2, recording_pb2
from bosdyn.api.mission import nodes_pb2
from bosdyn.client import ResponseError, RpcError
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.docking import DockingClient, docking_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.util import now_sec, seconds_to_timestamp

import robotMovements

LOGGER = logging.getLogger()

ASYNC_CAPTURE_RATE = 40  # milliseconds, 25 Hz
LINEAR_VELOCITY_DEFAULT = 0.6  # m/s
ANGULAR_VELOCITY_DEFAULT = 0.8  # rad/sec
COMMAND_DURATION_DEFAULT = 0.6  # seconds


def init_robot(hostname):
    """Initialize robot object"""

    # Initialize SDK
    sdk = bosdyn.client.create_standard_sdk('autonomousAutowalk', [bosdyn.mission.client.MissionClient])

    # Create robot object
    robot = sdk.create_robot(hostname)

    # Authenticate with robot
    bosdyn.client.util.authenticate(robot)

    # Establish time sync with the robot
    robot.time_sync.wait_for_sync()

    return robot

class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_robot_state_async()

def autonomous_autowalk():
    robotMovements._request_power_on()




class AutonomousAutowalk():
    def __init__(self, robot):
        self._robot = robot
        self._robot_id = self._robot.get_id()
        self._lease_keepalive = None
        self.walk = self._init_walk()
        self.directory = os.getcwd()

        # Initialize clients
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)

        # Clear graph to ensure only the data recorded using this example gets packaged into map
        self._graph_nav_client.clear_graph()
        self._recording_client = robot.ensure_client(
            GraphNavRecordingServiceClient.default_service_name)
        self._world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)
        self._docking_client = robot.ensure_client(DockingClient.default_service_name)

        # Initialize async tasks
        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        self._async_tasks = AsyncTasks([self._robot_state_task])
        self._async_tasks.update()

        # Timer for grabbing robot states
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._update_tasks)
        self.timer.start(ASYNC_CAPTURE_RATE)

        # Default starting speed values for the robot
        self.linear_velocity = LINEAR_VELOCITY_DEFAULT
        self.angular_velocity = ANGULAR_VELOCITY_DEFAULT
        self.command_duration = COMMAND_DURATION_DEFAULT

        # Experimentally determined default starting values for pitch, roll, yaw, and height
        self.euler_angles = geometry.EulerZXY()
        self.robot_height = 0

        self.resumed_recording = False
        self.elements = []
        self.fiducial_objects = []
        self.dock = None
        self.dock_and_end_recording = False




def main():
    """Record autowalk without GUI"""
    bosdyn.client.util.setup_logging()

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    args = parser.parse_args()

    # Initialize robot object
    robot = init_robot(args.hostname)
    assert not robot.is_estopped(), 'Robot is estopped. ' \
                                    'Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'
    auto_autowalk = AutonomousAutowalk(robot)




if __name__ == '__main__':
    if not main():
        sys.exit(1)
