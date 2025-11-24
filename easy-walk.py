import argparse
import os
import sys
import time
import math
from time import sleep

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api.basic_command_pb2 import SE2TrajectoryCommand
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.client import Robot
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.api.spot.robot_command_pb2 import MobilityParams
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.geometry import EulerZXY
from bosdyn.client.robot_command import RobotCommandBuilder

class RecordingInterface(object):
    def __init__(self, robot, download_filepath, client_metadata):
        self._download_filepath = os.path.join(download_filepath, 'downloaded_graph')
        self._recording_client = robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
        self._recording_environment = GraphNavRecordingServiceClient.make_recording_environment(
            waypoint_env=GraphNavRecordingServiceClient.make_waypoint_environment(client_metadata=client_metadata)
        )
        self._graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
        self._map_processing_client = robot.ensure_client(MapProcessingServiceClient.default_service_name)
        self._current_graph = None
        self._current_edges = dict()
        self._current_waypoint_snapshots = dict()
        self._current_edge_snapshots = dict()
        self._current_annotation_name_to_wp_id = dict()

    def should_we_start_recording(self):
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    return False
        return True

    def clear_map(self, *args):
        return self._graph_nav_client.clear_graph()

    def start_recording(self, *args):
        should_start_recording = self.should_we_start_recording()
        if not should_start_recording:
            print(
                'The system is not in the proper state to start recording.'
                'Try using the graph_nav_command_line to either clear the map or'
                'attempt to localize to the map.'
            )
            return
        try:
            status = self._recording_client.start_recording(recording_environment=self._recording_environment)
            print('Successfully started recording a map.')
        except Exception as err:
            print(f'Start recording failed: {err}')

    def stop_recording(self, *args):
        first_iter = True
        while True:
            try:
                status = self._recording_client.stop_recording()
                print('Successfully stopped recording a map.')
                break
            except bosdyn.client.recording.NotReadyYetError as err:
                if first_iter:
                    print('Cleaning up recording...')
                first_iter = False
                time.sleep(1.0)
                continue
            except Exception as err:
                print(f'Stop recording failed: {err}')
                break

    def get_recording_status(self, *args):
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print('The recording service is on.')
        else:
            print('The recording service is off.')

    def download_full_graph(self, *args):
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Failed to download the graph.')
            return
        self._write_full_graph(graph)
        print(f'Graph downloaded with {len(graph.waypoints)} waypoints and {len(graph.edges)} edges')
        self._download_and_write_waypoint_snapshots(graph.waypoints)
        self._download_and_write_edge_snapshots(graph.edges)

    def _write_full_graph(self, graph):
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, 'graph', graph_bytes)

    def _download_and_write_waypoint_snapshots(self, waypoints):
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(waypoint.snapshot_id)
            except Exception:
                print(f'Failed to download waypoint snapshot: {waypoint.snapshot_id}')
                continue
            self._write_bytes(
                os.path.join(self._download_filepath, 'waypoint_snapshots'),
                str(waypoint.snapshot_id),
                waypoint_snapshot.SerializeToString(),
            )
            num_waypoint_snapshots_downloaded += 1
            print(
                f'Downloaded {num_waypoint_snapshots_downloaded} of the total {len(waypoints)} waypoint snapshots.'
            )

    def _download_and_write_edge_snapshots(self, edges):
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                print(f'Failed to download edge snapshot: {edge.snapshot_id}')
                continue
            self._write_bytes(
                os.path.join(self._download_filepath, 'edge_snapshots'),
                str(edge.snapshot_id),
                edge_snapshot.SerializeToString(),
            )
            num_edge_snapshots_downloaded += 1
            print(
                f'Downloaded {num_edge_snapshots_downloaded} of the total {num_to_download} edge snapshots.'
            )

    def _write_bytes(self, filepath, filename, data):
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, filename), 'wb+') as f:
            f.write(data)
            f.close()

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print('Robot got lost when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print('Robot got stuck when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print('Robot is impaired.')
            return True
        else:
            return False

    def navigate_to(self):
        graph = self._graph_nav_client.download_graph()
        destination_waypoint = graph.waypoints[0]
        nav_to_cmd_id = None
        is_finished = False
        while not is_finished:
            nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint.id, 1, command_id=nav_to_cmd_id)
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            is_finished = self._check_success(nav_to_cmd_id)


def easy_walk(options):
    bosdyn.client.util.setup_logging(options.verbose)
    sdk = bosdyn.client.create_standard_sdk('easyWalk')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert not robot.is_estopped(), (
        'Robot is estopped. Please use an external E-Stop client, '
        'such as the estop SDK example, to configure E-Stop.'
    )
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    session_name = options.recording_session_name
    if session_name == '':
        session_name = os.path.basename(
            '/Users/gianmariagennai/Documents/Unifi/Magistrale/spot/autowalk/TestAuto'
        )
    user_name = options.recording_user_name
    if user_name == '':
        user_name = robot._current_user
    client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
        session_name=session_name,
        client_username=user_name,
        client_id='RecordingClient',
        client_type='Python SDK',
    )
    recordingInterface = RecordingInterface(robot, options.download_filepath, client_metadata)
    recordingInterface.stop_recording()
    recordingInterface.clear_map()
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')


        recordingInterface.start_recording()

        mobilityParams = MobilityParams()
        mobilityParams.vel_limit.max_vel.linear.x = 1.0
        mobilityParams.vel_limit.max_vel.linear.y = 0.5
        #mobilityParams.vel_limit.min_vel.angular = 0.5
        #mobilityParams.vel_limit.max_vel.angular = 1
        #mobilityParams.vel_limit.min_vel.angular = 1
        #mobilityParams.obstacle_params.obstacle_avoidance_padding = 0.1
        #mobilityParams.obstacle_params.disable_vision_foot_obstacle_avoidance = False
        #mobilityParams.obstacle_params.disable_vision_foot_obstacle_body_assist = False
        breakpoint()
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        command_client.robot_command(
            RobotCommandBuilder.synchro_trajectory_command_in_body_frame(0, 0, 3.14, robot.get_frame_tree_snapshot(),
                                                                         params=mobilityParams),
            end_time_secs=time.time() + 3)
        time.sleep(10)
        command_client.robot_command(
            RobotCommandBuilder.synchro_trajectory_command_in_body_frame(5, 0, 0, robot.get_frame_tree_snapshot(),
                                                                         params=mobilityParams),
            end_time_secs=time.time() + 10)
        time.sleep(5)
        command_client.robot_command(
            RobotCommandBuilder.synchro_trajectory_command_in_body_frame(0, 0, 1.5, robot.get_frame_tree_snapshot(),
                                                                         params=mobilityParams),
            end_time_secs=time.time() + 10)
        time.sleep(5)
        command_client.robot_command(
            RobotCommandBuilder.synchro_trajectory_command_in_body_frame(1, 0, 0, robot.get_frame_tree_snapshot(),
                                                                         params=mobilityParams),
            end_time_secs=time.time() + 10)
        robot.logger.info('Robot standing.')
        time.sleep(5)

        #tCommandBodyMov = {2}
        #bosdyn.client.robot_command.block_for_trajectory_cmd(command_client, id2,
                                                             # trajectory_end_statuses=tCommandStatus,
                                                             # body_movement_statuses=tCommandBodyMov, timeout_sec=30,
                                                             # logger=robot.logger)
        # command_client.robot_command(
        #     RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        #         21, 0, 0, robot.get_frame_tree_snapshot(), params=mobilityParams
        #     ),
        #     end_time_secs=time.time() + 20,
        # )
        #tCommandBodyMov = {2}
        # bosdyn.client.robot_command.block_for_trajectory_cmd(command_client, id2,
        #                                                      trajectory_end_statuses=tCommandStatus,
        #                                                      body_movement_statuses=tCommandBodyMov, timeout_sec=30,
        #                                                      logger=robot.logger)
        # sleep(5)
        #
        # id2 = command_client.robot_command(
        #     RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        #         0, 0, -(3.14 / 2), robot.get_frame_tree_snapshot(), params=mobilityParams
        #     ),
        #     end_time_secs=time.time() + 20,
        #)
        # tCommandBodyMov = {2}
        # bosdyn.client.robot_command.block_for_trajectory_cmd(command_client, id2,
        #                                                      trajectory_end_statuses=tCommandStatus,
        #                                                      body_movement_statuses=tCommandBodyMov, timeout_sec=30,
        #                                                      logger=robot.logger)
        # sleep(5)
        #
        # id2 = command_client.robot_command(
        #     RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
        #         12, 0, 0, robot.get_frame_tree_snapshot(), params=mobilityParams
        #     ),
        #     end_time_secs=time.time() + 20,
        # )
        # tCommandBodyMov = {2}
        # bosdyn.client.robot_command.block_for_trajectory_cmd(command_client, id2,
        #                                                      trajectory_end_statuses=tCommandStatus,
        #                                                      body_movement_statuses=tCommandBodyMov, timeout_sec=30,
        #                                                      logger=robot.logger)
        #sleep(5)

        robot.logger.info('Robot safely powered off.')
        log_comment = 'Easy autowalk.'
        robot.operator_comment(log_comment)
        robot.logger.info('Added comment "%s" to robot log.', log_comment)
        recordingInterface.stop_recording()
        recordingInterface.navigate_to()
        command_client.robot_command(RobotCommandBuilder.synchro_sit_command(), end_time_secs=time.time() + 20)
        sleep(1)
        robot.power_off(cut_immediately=False, timeout_sec=30)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')
        recordingInterface.download_full_graph()


def _maybe_display_image(image, display_time=3.0):
    try:
        import io
        from PIL import Image
    except ImportError:
        logger = bosdyn.client.util.get_logger()
        logger.warning("Missing dependencies. Can't display image.")
        return
    try:
        image = Image.open(io.BytesIO(image.data))
        image.show()
        time.sleep(display_time)
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.warning('Exception thrown displaying image. %r', exc)


def _maybe_save_image(image, path):
    logger = bosdyn.client.util.get_logger()
    try:
        import io
        from PIL import Image
    except ImportError:
        logger.warning("Missing dependencies. Can't save image.")
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


#TODO drifting problem when come back to first waypoint
def main():
    parser = argparse.ArgumentParser(description=None)
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        '-d',
        '--download-filepath',
        help='Full filepath for where to download graph and snapshots.',
        default=os.getcwd(),
    )
    parser.add_argument(
        '-n',
        '--recording_user_name',
        help=(
            'If a special user name should be attached to this session, use this name. '
            'If not provided, the robot username will be used.'
        ),
        default='',
    )
    parser.add_argument(
        '-s',
        '--recording_session_name',
        help=(
            'Provides a special name for this recording session. If not provided, '
            'the download filepath will be used.'
        ),
        default='',
    )
    options = parser.parse_args()
    try:
        easy_walk(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.error('Hello, Spot! threw an exception: %r', exc)
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
