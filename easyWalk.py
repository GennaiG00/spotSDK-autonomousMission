# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use the Boston Dynamics API"""

import argparse
import os
import sys
import time
import math

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api import trajectory_pb2, basic_command_pb2
from bosdyn.api.geometry_pb2 import SE2VelocityLimit, SE2Velocity, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration
from bosdyn.api.spot.robot_command_pb2 import MobilityParams
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient



class RecordingInterface(object):
    def __init__(self, robot, download_filepath, client_metadata):
        # Filepath for the location to put the downloaded graph and snapshots.
        self._download_filepath = os.path.join(download_filepath, 'downloaded_graph')

        # Set up the recording service client.
        self._recording_client = robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)

        # Create the recording environment.
        self._recording_environment = GraphNavRecordingServiceClient.make_recording_environment(waypoint_env=GraphNavRecordingServiceClient.make_waypoint_environment(client_metadata=client_metadata))

        # Set up the graph nav service client.
        self._graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)

        self._map_processing_client = robot.ensure_client(MapProcessingServiceClient.default_service_name)

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  # maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

    def should_we_start_recording(self):
        # Before starting to record, check the state of the GraphNav system.
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            # Check that the graph has waypoints. If it does, then we need to be localized to the graph
            # before starting to record
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    # Not localized to anything in the map. The best option is to clear the graph or
                    # attempt to localize to the current map.
                    # Returning false since the GraphNav system is not in the state it should be to
                    # begin recording.
                    return False
        # If there is no graph or there exists a graph that we are localized to, then it is fine to
        # start recording, so we return True.
        return True

    def clear_map(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def start_recording(self, *args):
        """Start recording a map."""
        should_start_recording = self.should_we_start_recording()
        if not should_start_recording:
            print('The system is not in the proper state to start recording.'
                  'Try using the graph_nav_command_line to either clear the map or'
                  'attempt to localize to the map.')
            return
        try:
            status = self._recording_client.start_recording(
                recording_environment=self._recording_environment)
            print('Successfully started recording a map.')
        except Exception as err:
            print(f'Start recording failed: {err}')

    def stop_recording(self, *args):
        """Stop or pause recording a map."""
        first_iter = True
        while True:
            try:
                status = self._recording_client.stop_recording()
                print('Successfully stopped recording a map.')
                break
            except bosdyn.client.recording.NotReadyYetError as err:
                # It is possible that we are not finished recording yet due to
                # background processing. Try again every 1 second.
                if first_iter:
                    print('Cleaning up recording...')
                first_iter = False
                time.sleep(1.0)
                continue
            except Exception as err:
                print(f'Stop recording failed: {err}')
                break

    def get_recording_status(self, *args):
        """Get the recording service's status."""
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print('The recording service is on.')
        else:
            print('The recording service is off.')

    def download_full_graph(self, *args):
        """Download the graph and snapshots from the robot."""
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Failed to download the graph.')
            return
        self._write_full_graph(graph)
        print(
            f'Graph downloaded with {len(graph.waypoints)} waypoints and {len(graph.edges)} edges')
        # Download the waypoint and edge snapshots.
        self._download_and_write_waypoint_snapshots(graph.waypoints)
        self._download_and_write_edge_snapshots(graph.edges)

    def _write_full_graph(self, graph):
        """Download the graph from robot to the specified, local filepath location."""
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, 'graph', graph_bytes)

    def _download_and_write_waypoint_snapshots(self, waypoints):
        """Download the waypoint snapshots from robot to the specified, local filepath location."""
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(
                    waypoint.snapshot_id)
            except Exception:
                # Failure in downloading waypoint snapshot. Continue to next snapshot.
                print(f'Failed to download waypoint snapshot: {waypoint.snapshot_id}')
                continue
            self._write_bytes(os.path.join(self._download_filepath, 'waypoint_snapshots'),
                              str(waypoint.snapshot_id), waypoint_snapshot.SerializeToString())
            num_waypoint_snapshots_downloaded += 1
            print(
                f'Downloaded {num_waypoint_snapshots_downloaded} of the total {len(waypoints)} waypoint snapshots.'
            )

    def _download_and_write_edge_snapshots(self, edges):
        """Download the edge snapshots from robot to the specified, local filepath location."""
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                # Failure in downloading edge snapshot. Continue to next snapshot.
                print(f'Failed to download edge snapshot: {edge.snapshot_id}')
                continue
            self._write_bytes(os.path.join(self._download_filepath, 'edge_snapshots'),
                              str(edge.snapshot_id), edge_snapshot.SerializeToString())
            num_edge_snapshots_downloaded += 1
            print(
                f'Downloaded {num_edge_snapshots_downloaded} of the total {num_to_download} edge snapshots.'
            )

    def _write_bytes(self, filepath, filename, data):
        """Write data to a file."""
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, filename), 'wb+') as f:
            f.write(data)
            f.close()


def hello_spot(options):
    # The Boston Dynamics Python library uses Python's logging module to
    # generate output. Applications using the library can specify how
    # the logging information should be output.
    bosdyn.client.util.setup_logging(options.verbose)

    # The SDK object is the primary entry point to the Boston Dynamics API.
    # create_standard_sdk will initialize an SDK object with typical default
    # parameters. The argument passed in is a string identifying the client.
    sdk = bosdyn.client.create_standard_sdk('easyWalk')

    # A Robot object represents a single robot. Clients using the Boston
    # Dynamics API can manage multiple robots, but this tutorial limits
    # access to just one. The network address of the robot needs to be
    # specified to reach it. This can be done with a DNS name
    # (e.g. spot.intranet.example.com) or an IP literal (e.g. 10.0.63.1)
    robot = sdk.create_robot(options.hostname)

    # Clients need to authenticate to a robot before being able to use it.
    bosdyn.client.util.authenticate(robot)

    # Establish time sync with the robot. This kicks off a background thread to establish time sync.
    # Time sync is required to issue commands to the robot. After starting time sync thread, block
    # until sync is established.
    robot.time_sync.wait_for_sync()

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    # The robot state client will allow us to get the robot's state information, and construct
    # a command using frame information published by the robot.
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    # Only one client at a time can operate a robot. Clients acquire a lease to
    # indicate that they want to control a robot. Acquiring may fail if another
    # client is currently controlling the robot. When the client is done
    # controlling the robot, it should return the lease so other clients can
    # control it. The LeaseKeepAlive object takes care of acquiring and returning
    # the lease for us.
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    # Parse session and user name options.
    session_name = options.recording_session_name
    if session_name == '':
        session_name = os.path.basename(options.download_filepath)
    user_name = options.recording_user_name
    if user_name == '':
        user_name = robot._current_user

    # Crate metadata for the recording session.
    client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
        session_name=session_name, client_username=user_name, client_id='RecordingClient', client_type='Python SDK')

    recordingInterface = RecordingInterface(robot, options.download_filepath, client_metadata)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        recordingInterface.start_recording()

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.

        mobilityParams = MobilityParams()
        mobilityParams.vel_limit.max_vel.linear.x = 1
        mobilityParams.vel_limit.max_vel.linear.y = 0.5
        #mobilityParams.vel_limit.max_vel.angular = math.pi/8

        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        command_client.robot_command(
            RobotCommandBuilder.synchro_trajectory_command_in_body_frame(0, 0, 3.14, robot.get_frame_tree_snapshot()),
            end_time_secs=time.time() + 60)
        time.sleep(5)
        command_client.robot_command(RobotCommandBuilder.synchro_trajectory_command_in_body_frame(10.0, 0, 0,robot.get_frame_tree_snapshot(), params=mobilityParams), end_time_secs=time.time() + 60)
        time.sleep(5)
        command_client.robot_command(RobotCommandBuilder.synchro_trajectory_command_in_body_frame(0, 0, 3.14,robot.get_frame_tree_snapshot()), end_time_secs=time.time() + 60)
        time.sleep(5)
        command_client.robot_command(RobotCommandBuilder.synchro_trajectory_command_in_body_frame(10.0, 0, 0,robot.get_frame_tree_snapshot(), params=mobilityParams), end_time_secs=time.time() + 60)
        time.sleep(5)
        robot.logger.info('Robot standing.')


        #robot.power_off(cut_immediately=False, timeout_sec=20)

        # Query the robot for its current state before issuing the stand with yaw command.
        # This state prov22ides a reference pose for issuing a frame based body offset command.
        robot_state = robot_state_client.get_robot_state()

        # Tell the robot to stand in a twisted position.
        #
        # The RobotCommandBuilder constructs command messages, which are then
        # issued to the robot using "robot_command" on the command client.
        #
        # In this example, the RobotCommandBuilder generates a stand command
        # message with a non-default rotation in the footprint frame. The footprint
        # frame is a gravity aligned frame with its origin located at the geometric
        # center of the feet. The X axis of the footprint frame points forward along
        # the robot's length, the Z axis points up aligned with gravity, and the Y
        # axis is the cross-product of the two.
        # footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.4, roll=0.0, pitch=0.0)
        # cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
        # command_client.robot_command(cmd)
        # robot.logger.info('Robot standing twisted.')
        # time.sleep(3)

        # Now compute an absolute desired position and orientation of the robot body origin.
        # Use the frame helper class to compute the world to gravity aligned body frame transformation.
        # Note, the robot_state used here was cached from before the above yaw stand command,
        # so it contains the nominal stand pose.
        # odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
        #                                  ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        #
        # # Specify a trajectory to shift the body forward followed by looking down, then return to nominal.
        # # Define times (in seconds) for each point in the trajectory.
        # t1 = 2.5
        # t2 = 5.0
        # t3 = 7.5
        #
        # # Specify the poses as transformations to the cached flat_body pose.
        # flat_body_T_pose2 = math_helpers.SE3Pose(x=0.0, y=0, z=0, rot=math_helpers.Quat(w=0.9848, x=0, y=0.1736, z=0))
        # flat_body_T_pose3 = math_helpers.SE3Pose(x=0.0, y=0, z=0, rot=math_helpers.Quat())
        #
        # traj_point2 = trajectory_pb2.SE3TrajectoryPoint(pose=(odom_T_flat_body * flat_body_T_pose2).to_proto(), time_since_reference=seconds_to_duration(t2))
        # traj_point3 = trajectory_pb2.SE3TrajectoryPoint(pose=(odom_T_flat_body * flat_body_T_pose3).to_proto(), time_since_reference=seconds_to_duration(t3))
        #
        # # Build the trajectory proto by combining the points.
        # traj = trajectory_pb2.SE3Trajectory(points=[traj_point1, traj_point2, traj_point3])
        #
        # # Build a custom mobility params to specify absolute body control.
        # body_control = spot_command_pb2.BodyControlParams(
        #     body_pose=spot_command_pb2.BodyControlParams.BodyPose(root_frame_name=ODOM_FRAME_NAME,
        #                                                           base_offset_rt_root=traj))
        #
        # # Issue the command via the RobotCommandClient
        # robot.logger.info('Beginning absolute body control while standing.')
        # blocking_stand(command_client, timeout_sec=10,
        #                params=spot_command_pb2.MobilityParams(body_control=body_control))
        # robot.logger.info('Finished absolute body control while standing.')

        # Capture an image.
        # Spot has five sensors around the body. Each sensor consists of a stereo pair and a
        # fisheye camera. The list_image_sources RPC gives a list of image sources which are
        # available to the API client. Images are captured via calls to the get_image RPC.
        # Images can be requested from multiple image sources in one call.
        # image_client = robot.ensure_client(ImageClient.default_service_name)
        # sources = image_client.list_image_sources()
        # image_response = image_client.get_image_from_sources(['frontleft_fisheye_image'])
        # _maybe_display_image(image_response[0].shot.image)
        # if config.save or config.save_path is not None:
        #     _maybe_save_image(image_response[0].shot.image, config.save_path)

        # Log a comment.
        # Comments logged via this API are written to the robots test log. This is the best way
        # to mark a log as "interesting". These comments will be available to Boston Dynamics
        # devs when diagnosing customer issues.
        log_comment = 'Easy autowalk.'
        robot.operator_comment(log_comment)
        robot.logger.info('Added comment "%s" to robot log.', log_comment)

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.

        recordingInterface.stop_recording()

        robot.power_off(cut_immediately=False, timeout_sec=30)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')

        recordingInterface.download_full_graph()


def _maybe_display_image(image, display_time=3.0):
    """Try to display image, if client has correct deps."""
    try:
        import io

        from PIL import Image
    except ImportError:
        logger = bosdyn.client.util.get_logger()
        logger.warning('Missing dependencies. Can\'t display image.')
        return
    try:
        image = Image.open(io.BytesIO(image.data))
        image.show()
        time.sleep(display_time)
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.warning('Exception thrown displaying image. %r', exc)


def _maybe_save_image(image, path):
    """Try to save image, if client has correct deps."""
    logger = bosdyn.client.util.get_logger()
    try:
        import io

        from PIL import Image
    except ImportError:
        logger.warning('Missing dependencies. Can\'t save image.')
        return
    name = 'hello-spot-img.jpg'
    if path is not None and os.path.exists(path):
        path = os.path.join(os.getcwd(), path)
        name = os.path.join(path, name)
        logger.info('Saving image to: %s', name)
    else:
        logger.info('Saving image to working directory as %s', name)
    try:
        image = Image.open(io.BytesIO(image.data))
        image.save(name)
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.warning('Exception thrown saving image. %r', exc)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-d', '--download-filepath',
                        help='Full filepath for where to download graph and snapshots.',
                        default=os.getcwd())
    parser.add_argument(
        '-n', '--recording_user_name', help=
        'If a special user name should be attached to this session, use this name. If not provided, the robot username will be used.',
        default='')
    parser.add_argument(
        '-s', '--recording_session_name', help=
        'Provides a special name for this recording session. If not provided, the download filepath will be used.',
        default='')

    options = parser.parse_args()
    try:
        hello_spot(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.error('Hello, Spot! threw an exception: %r', exc)
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
