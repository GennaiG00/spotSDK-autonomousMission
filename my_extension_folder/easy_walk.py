import argparse
import math
import os
import sys
import time
from time import sleep
import numpy as np
from bosdyn.api.graph_nav.graph_nav_pb2 import TravelParams

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api.graph_nav import graph_nav_pb2, recording_pb2, map_processing_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client.graph_nav import GraphNavClient, map_pb2
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client import math_helpers
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.api import local_grid_pb2
from types import SimpleNamespace
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from google.protobuf import wrappers_pb2 as wrappers


#---------------- OBSTACLE GRID START -----------------------

def create_vtk_no_step_grid(proto, robot_state_client):
    """Generate VTK polydata for the no step grid from the local grid response."""
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'no_step':
            local_grid_proto = local_grid_found
            cell_size = local_grid_found.local_grid.extent.cell_size
    # Unpack the data field for the local grid.
    cells_no_step = unpack_grid(local_grid_proto).astype(np.float32)
    # Populate the x,y values with a complete combination of all possible pairs for the dimensions in the grid extent.
    ys, xs = np.mgrid[0:local_grid_proto.local_grid.extent.num_cells_x,
                      0:local_grid_proto.local_grid.extent.num_cells_y]
    # Get the estimated height (z value) of the ground in the vision frame as if the robot was standing.
    transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
    vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    z_ground_in_vision_frame = compute_ground_height_in_vision_frame(robot_state_client)
    # Numpy vstack makes it so that each column is (x,y,z) for a single no step grid point. The height values come
    # from the estimated height of the ground plane.
    cell_count = local_grid_proto.local_grid.extent.num_cells_x * local_grid_proto.local_grid.extent.num_cells_y
    cells_est_height = np.ones(cell_count) * z_ground_in_vision_frame
    pts = np.vstack(
        [np.ravel(xs).astype(np.float32),
         np.ravel(ys).astype(np.float32), cells_est_height]).T
    pts[:, [0, 1]] *= (local_grid_proto.local_grid.extent.cell_size,
                       local_grid_proto.local_grid.extent.cell_size)
    # Determine the coloration based on whether or not the region is steppable. The regions that Spot considers it
    # cannot safely step are colored red, and the regions that are considered safe to step are colored blue.
    color = np.zeros([cell_count, 3], dtype=np.uint8)
    color[:, 0] = (cells_no_step <= 0.0)
    color[:, 2] = (cells_no_step > 0.0)
    color *= 255
    # Offset the grid points to be in the vision frame instead of the local grid frame.
    vision_tform_local_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME,
                                            local_grid_proto.local_grid.frame_name_local_grid_data)
    pts = offset_grid_pixels(pts, vision_tform_local_grid, cell_size)

    return pts, cells_no_step, color

def compute_ground_height_in_vision_frame(robot_state_client):
    """Get the z-height of the ground plane in vision frame from the current robot state."""
    robot_state = robot_state_client.get_robot_state()
    vision_tform_ground_plane = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                              VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    return vision_tform_ground_plane.position.z

def offset_grid_pixels(pts, vision_tform_local_grid, cell_size):
    """Offset the local grid's pixels to be in the world frame instead of the local grid frame."""
    x_base = vision_tform_local_grid.position.x + cell_size * 0.5
    y_base = vision_tform_local_grid.position.y + cell_size * 0.5
    pts[:, 0] += x_base
    pts[:, 1] += y_base
    return pts

def unpack_grid(local_grid_proto):
    """Unpack the local grid proto."""
    # Determine the data type for the bytes data.
    data_type = get_numpy_data_type(local_grid_proto.local_grid)
    if data_type is None:
        print('Cannot determine the dataformat for the local grid.')
        return None
    # Decode the local grid.
    if local_grid_proto.local_grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RAW:
        full_grid = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
    elif local_grid_proto.local_grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RLE:
        full_grid = expand_data_by_rle_count(local_grid_proto, data_type=data_type)
    else:
        # Return nothing if there is no encoding type set.
        return None
    # Apply the offset and scaling to the local grid.
    if local_grid_proto.local_grid.cell_value_scale == 0:
        return full_grid
    full_grid_float = full_grid.astype(np.float64)
    full_grid_float *= local_grid_proto.local_grid.cell_value_scale
    full_grid_float += local_grid_proto.local_grid.cell_value_offset
    return full_grid_float

def get_numpy_data_type(local_grid_proto):
    """Convert the cell format of the local grid proto to a numpy data type."""
    if local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_UINT16:
        return np.uint16
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_INT16:
        return np.int16
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_UINT8:
        return np.uint8
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_INT8:
        return np.int8
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT64:
        return np.float64
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT32:
        return np.float32
    else:
        return None

def expand_data_by_rle_count(local_grid_proto, data_type=np.int16):
    """Expand local grid data to full bytes data using the RLE count."""
    cells_pz = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
    cells_pz_full = []
    # For each value of rle_counts, we expand the cell data at the matching index
    # to have that many repeated, consecutive values.
    for i in range(0, len(local_grid_proto.local_grid.rle_counts)):
        for j in range(0, local_grid_proto.local_grid.rle_counts[i]):
            cells_pz_full.append(cells_pz[i])
    return np.array(cells_pz_full)

def analyze_navigation_zones(pts, cells_no_step, robot_x, robot_y, robot_yaw,
                            front_distance=0.5, lateral_distance=1.5, lateral_width=1.0):
    """    Analyze zones in front, to the right and to the left of the robot to determine whether the path is clear.

    Args:
        pts: array (N, 3) with coordinates [x, y, z] of the cells in the VISION frame
        cells_no_step: array (N,) with no-step values (<=0 = non-steppable, >0 = steppable)
        robot_x, robot_y: robot position in meters (VISION frame)
        robot_yaw: robot orientation in radians
        front_distance: how far to look ahead (meters)
        lateral_distance: how far to look sideways (meters)
        lateral_width: width of the lateral area to consider (meters)

    Returns:
        dict with keys:
            - 'front_blocked': bool, True if front is blocked
            - 'left_free': bool, True if left side is free
            - 'right_free': bool, True if right side is free
            - 'front_free_ratio': float 0-1, fraction of free cells in front
            - 'left_free_ratio': float 0-1, fraction of free cells on the left
            - 'right_free_ratio': float 0-1, fraction of free cells on the right
            - 'recommendation': str, suggestion ('GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'BLOCKED')
    """

    # Direction vectors for the robot (body X-axis in the VISION frame)
    front_dir_x = np.cos(robot_yaw)
    front_dir_y = np.sin(robot_yaw)

    # Perpendicular direction vectors (right and left)
    right_dir_x = np.cos(robot_yaw - np.pi/2)  # -90° = right
    right_dir_y = np.sin(robot_yaw - np.pi/2)
    left_dir_x = np.cos(robot_yaw + np.pi/2)   # +90° = left
    left_dir_y = np.sin(robot_yaw + np.pi/2)

    # Coordinates of grid cells relative to the robot
    dx = pts[:, 0] - robot_x
    dy = pts[:, 1] - robot_y

    # Projection onto the forward direction (longitudinal distance)
    proj_front = dx * front_dir_x + dy * front_dir_y
    # Projection onto the lateral direction (transverse distance, + = right, - = left)
    proj_lateral = dx * right_dir_x + dy * right_dir_y

    # --- FRONT ZONE ---
    # Cells in front of the robot: proj_front > 0 and < front_distance, |proj_lateral| < 0.5m (robot width ~0.8m)
    mask_front = (proj_front > 0) & (proj_front <= front_distance) & (np.abs(proj_lateral) <= 0.5)
    cells_front = cells_no_step[mask_front]

    if len(cells_front) > 0:
        front_free_count = np.sum(cells_front > 0.0)
        front_free_ratio = front_free_count / len(cells_front)
    else:
        front_free_ratio = 1.0  # no cells = consider free

    # Threshold: if less than 70% of cells are free, consider the path blocked
    front_blocked = front_free_ratio < 0.7

    # --- LEFT ZONE ---
    # Cells to the left: proj_lateral < 0 (left), proj_front between 0 and lateral_distance, |proj_lateral| < lateral_width
    mask_left = (proj_front > 0) & (proj_front <= lateral_distance) & \
                (proj_lateral < 0) & (proj_lateral >= -lateral_width)
    cells_left = cells_no_step[mask_left]

    if len(cells_left) > 0:
        left_free_count = np.sum(cells_left > 0.0)
        left_free_ratio = left_free_count / len(cells_left)
    else:
        left_free_ratio = 1.0

    left_free = left_free_ratio > 0.7

    # --- RIGHT ZONE ---
    # Cells to the right: proj_lateral > 0 (right), proj_front between 0 and lateral_distance, proj_lateral < lateral_width
    mask_right = (proj_front > 0) & (proj_front <= lateral_distance) & \
                 (proj_lateral > 0) & (proj_lateral <= lateral_width)
    cells_right = cells_no_step[mask_right]

    if len(cells_right) > 0:
        right_free_count = np.sum(cells_right > 0.0)
        right_free_ratio = right_free_count / len(cells_right)
    else:
        right_free_ratio = 1.0

    right_free = right_free_ratio > 0.7

    # --- RECOMMENDATION ---
    if not front_blocked:
        recommendation = 'GO_STRAIGHT'
    elif left_free and right_free:
        # Both sides are free: choose the one with more space
        recommendation = 'TURN_LEFT' if left_free_ratio >= right_free_ratio else 'TURN_RIGHT'
    elif left_free:
        recommendation = 'TURN_LEFT'
    elif right_free:
        recommendation = 'TURN_RIGHT'
    else:
        recommendation = 'BLOCKED'

    return {
        'front_blocked': front_blocked,
        'left_free': left_free,
        'right_free': right_free,
        'front_free_ratio': front_free_ratio,
        'left_free_ratio': left_free_ratio,
        'right_free_ratio': right_free_ratio,
        'recommendation': recommendation,
        'masks': {  # for debug/visualization
            'front': mask_front,
            'left': mask_left,
            'right': mask_right
        }
    }

#---------------- OBSTACLE GRID END-----------------------

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
        self.robot = robot

    def _get_transform(self, from_wp, to_wp):
        """Get transform from from-waypoint to to-waypoint."""

        from_se3 = from_wp.waypoint_tform_ko
        from_tf = SE3Pose(
            from_se3.position.x, from_se3.position.y, from_se3.position.z,
            Quat(w=from_se3.rotation.w, x=from_se3.rotation.x, y=from_se3.rotation.y,
                 z=from_se3.rotation.z))

        to_se3 = to_wp.waypoint_tform_ko
        to_tf = SE3Pose(
            to_se3.position.x, to_se3.position.y, to_se3.position.z,
            Quat(w=to_se3.rotation.w, x=to_se3.rotation.x, y=to_se3.rotation.y,
                 z=to_se3.rotation.z))

        from_T_to = from_tf.mult(to_tf.inverse())
        return from_T_to.to_proto()

    def create_new_edge(self):
        graph = self._graph_nav_client.download_graph()
        if len(graph.waypoints) < 2:
            print(f'Graph contains {len(graph.waypoints)} waypoints -- at least two are needed to create loop.')
            return False

        first_waypoint = None
        for waypoint in graph.waypoints:
            if waypoint.annotations.name == "waypoint_0":
                first_waypoint = waypoint
        if first_waypoint is None:
            print('No waypoint_0 found in the aph.')
            return False

        from_wp = first_waypoint
        if from_wp is None:
            return

        to_wp = max(graph.waypoints, key=lambda wp: int(wp.annotations.name.split('_')[-1]))
        if to_wp is None:
            return

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_wp.id
        new_edge.id.to_waypoint = to_wp.id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        print(f'edge transform = {new_edge.from_tform_to}')

        # Send request to add edge to map
        self._recording_client.create_edge(edge=new_edge)

    def should_we_start_recording(self):
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    return False
        return True

    def create_default_waypoint(self):
        """Create a waypoint with an incremental ID (e.g., waypoint_0, waypoint_1)."""
        graph = self._graph_nav_client.download_graph()
        if not graph.waypoints:
            next_number = 0
        else:
            try:
                last_wp = max(graph.waypoints, key=lambda wp: int(wp.annotations.name.split('_')[-1]))
                last_number = int(last_wp.annotations.name.split('_')[-1])
                next_number = last_number + 1
            except (ValueError, IndexError):
                print("Could not parse existing waypoint names to determine next number.")
                return False

        new_name = f'waypoint_{next_number}'
        print(f"Creating waypoint with name: {new_name}")
        resp = self._recording_client.create_waypoint(waypoint_name=new_name)

        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            print('Successfully created a waypoint.')
        else:
            print('Could not create a waypoint.')

    def get_recording_status(self, *args):
        """Get the recording service's status."""
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print('The recording service is on.')
            print(status)
        else:
            print('The recording service is off.')

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
                print('Successfully stopped recording a map.' + status)
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

    def navigate_to_first_waypoint(self):
        """Navigate back to the first waypoint (waypoint_0)."""
        graph = self._graph_nav_client.download_graph()
        first_waypoint = None
        for waypoint in graph.waypoints:
            if waypoint.annotations.name == "waypoint_0":
                first_waypoint = waypoint
        if first_waypoint is None:
            print('No waypoint_0 found in the graph.')
            return False

        print(f"🔙 Navigating back to first waypoint (waypoint_0)...")
        nav_to_cmd_id = None
        is_finished = False

        travel_params = TravelParams()
        travel_params.lost_detector_strictness = 2



        while not is_finished:
            nav_to_cmd_id = self._graph_nav_client.navigate_to(first_waypoint.id, 1.0, command_id=nav_to_cmd_id)
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            is_finished = self._check_success(nav_to_cmd_id)

        print(f"✅ Arrived at first waypoint")
        return True

    def navigate_to_previous_waypoint(self, steps_back=3):
        """
        Navigate back to a previous waypoint (n steps back from the last recorded waypoint).

        This is used when the robot's native collision avoidance fails and we want to retry
        with custom obstacle avoidance logic.

        Args:
            steps_back: Number of waypoints to go back (default 3-4 for safety margin)

        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        graph = self._graph_nav_client.download_graph()

        if not graph or len(graph.waypoints) == 0:
            print('⚠️ No waypoints found in the graph.')
            return False

        # Sort waypoints by name (assuming they are named waypoint_0, waypoint_1, etc.)
        sorted_waypoints = sorted(graph.waypoints, key=lambda wp: wp.annotations.name)

        # Find the target waypoint (steps_back from the end)
        total_waypoints = len(sorted_waypoints)

        if total_waypoints <= steps_back:
            # If we don't have enough waypoints, go back to the first one
            target_index = 0
            print(f'⚠️ Not enough waypoints to go back {steps_back} steps. Going to first waypoint instead.')
        else:
            # Go back steps_back waypoints from the last one
            target_index = total_waypoints - steps_back - 1

        target_waypoint = sorted_waypoints[target_index]
        target_name = target_waypoint.annotations.name

        print(f"🔙 Navigating back to previous waypoint: {target_name} (going back {steps_back} waypoints)")
        print(f"   Total waypoints in graph: {total_waypoints}")
        print(f"   Target waypoint index: {target_index}")

        # Navigate to the target waypoint
        nav_to_cmd_id = None
        is_finished = False

        while not is_finished:
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    target_waypoint.id, 1.0, command_id=nav_to_cmd_id
                )
            except Exception as e:
                print(f'❌ Error while navigating to {target_name}: {e}')
                return False

            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            is_finished = self._check_success(nav_to_cmd_id)

        print(f"✅ Arrived at waypoint {target_name}")
        print(f"🚀 Ready to retry with custom obstacle avoidance")
        return True


def relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client, stairs=False):
    """Move the robot relative to its current pose.

    Returns:
        tuple: (success: bool, distance_traveled: float)
               - success: True if goal reached, False if failed
               - distance_traveled: meters traveled before stopping/failing
    """
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Save initial position
    initial_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    initial_x = initial_tform_body.x
    initial_y = initial_tform_body.y

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame.
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 6000.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)

    # Wait until the robot has reached the goal or fails
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback

        # Get current position
        current_state = robot_state_client.get_robot_state()
        current_transforms = current_state.kinematic_state.transforms_snapshot
        current_tform_body = get_se2_a_tform_b(current_transforms, frame_name, BODY_FRAME_NAME)

        # Calculate distance traveled
        distance_traveled = np.sqrt((current_tform_body.x - initial_x) ** 2 +
                                    (current_tform_body.y - initial_y) ** 2)

        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print(f'Failed to reach the goal (traveled {distance_traveled:.2f}m)')
            return False, distance_traveled

        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print(f'Arrived at the goal (traveled {distance_traveled:.2f}m)')
            return True, distance_traveled

        time.sleep(1)


def check_path_clear(local_grid_client, robot_state_client, front_distance=0.2, threshold=0.7):
    """
    Check whether the path in front of the robot is clear.
    Uses the transforms_snapshot from the local grid to ensure synchronization
    between robot position and grid data.

    Returns:
        dict with 'front_blocked', 'left_free', 'right_free', 'recommendation'
    """
    # Get local grid (this contains a transforms_snapshot at the time the grid was captured)
    proto = local_grid_client.get_local_grids(['no_step'])
    pts, cells_no_step, color = create_vtk_no_step_grid(proto, robot_state_client)

    # Extract the local grid proto to access its transforms_snapshot
    local_grid_proto = None
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'no_step':
            local_grid_proto = local_grid_found
            break

    if local_grid_proto is None:
        print("⚠️ No 'no_step' grid found in response")
        # Return safe defaults
        return {
            'front_blocked': False,
            'left_free': True,
            'right_free': True,
            'front_free_ratio': 1.0,
            'left_free_ratio': 1.0,
            'right_free_ratio': 1.0,
            'recommendation': 'GO_STRAIGHT',
            'masks': {'front': [], 'left': [], 'right': []}
        }

    transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot

    # Get robot position using the SAME transforms_snapshot as the grid
    try:
        vision_tform_body = get_a_tform_b(
            transforms_snapshot,  # Use grid's snapshot, not current robot state
            VISION_FRAME_NAME,
            BODY_FRAME_NAME
        )
    except Exception as e:
        print(f"⚠️ Failed to get robot transform from grid snapshot: {e}")
        # Fallback: use current robot state (less accurate but better than crashing)
        robot_state = robot_state_client.get_robot_state()
        vision_tform_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            VISION_FRAME_NAME,
            BODY_FRAME_NAME
        )

    robot_x = vision_tform_body.position.x
    robot_y = vision_tform_body.position.y
    quat = vision_tform_body.rotation
    robot_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                           1.0 - 2.0 * (quat.y**2 + quat.z**2))

    # Analyze zones (now synchronized!)
    nav_analysis = analyze_navigation_zones(pts, cells_no_step, robot_x, robot_y, robot_yaw,
                                           front_distance=front_distance,
                                           lateral_distance=1.5,
                                           lateral_width=1.0)

    return nav_analysis


def safe_relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client,
                       local_grid_client, recording_interface, lateral_offset=2.0, max_retries=2):
    """
    Move the robot with integrated obstacle avoidance.

    Returns:
        tuple: (success: bool, total_distance: float)
    """
    print(f"\n{'=' * 60}")
    print(f"🎯 REQUESTED MOVE: dx={dx:.2f}m, dy={dy:.2f}m, dyaw={np.rad2deg(dyaw):.1f}°")
    print(f"{'=' * 60}")

    # Save initial position AND orientation
    initial_state = robot_state_client.get_robot_state()
    initial_vision_tform_body = get_a_tform_b(
        initial_state.kinematic_state.transforms_snapshot,
        VISION_FRAME_NAME,
        BODY_FRAME_NAME
    )
    initial_x = initial_vision_tform_body.position.x
    initial_y = initial_vision_tform_body.position.y

    quat = initial_vision_tform_body.rotation
    initial_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                             1.0 - 2.0 * (quat.y ** 2 + quat.z ** 2))

    # ⭐ Calculate target yaw (initial + requested rotation)
    target_yaw = initial_yaw + dyaw
    # Normalize to [-π, π]
    target_yaw = np.arctan2(np.sin(target_yaw), np.cos(target_yaw))

    print(f"📍 Mission start state:")
    print(f"   Position: ({initial_x:.2f}, {initial_y:.2f}) m")
    print(f"   Initial yaw: {np.rad2deg(initial_yaw):.1f}°")
    print(f"   Target yaw: {np.rad2deg(target_yaw):.1f}°")

    total_distance = abs(dx)
    total_traveled = 0.0

    # Try main movement
    success, distance_traveled = relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client)
    total_traveled += distance_traveled

    # ⭐ Check orientation after movement (even if success=True)
    current_state = robot_state_client.get_robot_state()
    current_vision_tform_body = get_a_tform_b(
        current_state.kinematic_state.transforms_snapshot,
        VISION_FRAME_NAME,
        BODY_FRAME_NAME
    )
    current_quat = current_vision_tform_body.rotation
    current_yaw = np.arctan2(2.0 * (current_quat.w * current_quat.z + current_quat.x * current_quat.y),
                             1.0 - 2.0 * (current_quat.y ** 2 + current_quat.z ** 2))

    # Calculate yaw error
    yaw_error = target_yaw - current_yaw
    # Normalize to [-π, π]
    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

    print(f"\n📐 Orientation check:")
    print(f"   Current yaw: {np.rad2deg(current_yaw):.1f}°")
    print(f"   Target yaw: {np.rad2deg(target_yaw):.1f}°")
    print(f"   Error: {np.rad2deg(yaw_error):.1f}°")

    # ⭐ If orientation is wrong (> 5°), correct it EVEN IF movement succeeded
    orientation_tolerance = np.deg2rad(5)  # 5° tolerance

    if abs(yaw_error) > orientation_tolerance:
        print(f"⚠️ Orientation error detected ({np.rad2deg(yaw_error):.1f}°)")
        print(f"🔄 Correcting orientation before proceeding...")

        restore_success, _ = relative_move(0, 0, yaw_error, frame_name,
                                           robot_command_client, robot_state_client)
        if restore_success:
            print(f"✅ Orientation corrected")
            # ⭐ If movement succeeded AND orientation was corrected → SUCCESS
            if success:
                print(f"✅ MOVEMENT COMPLETE: {total_traveled:.2f}m traveled successfully!")
                return True, total_traveled
        else:
            print(f"❌ Failed to restore orientation")
            success = False  # Mark as failure if we can't restore orientation

    # ⭐ If movement succeeded AND orientation is correct → SUCCESS
    if success:
        print(f"✅ MOVEMENT COMPLETE: {total_traveled:.2f}m traveled successfully!")
        return True, total_traveled

    # ⚠️ Movement failed or orientation can't be corrected → GO BACK
    print(f"❌ Movement failed after {distance_traveled:.2f}m")
    print(f"🔄 Restoring initial orientation before going back...")

    # Restore initial orientation (not target, but starting orientation)
    yaw_diff_to_initial = initial_yaw - current_yaw
    yaw_diff_to_initial = np.arctan2(np.sin(yaw_diff_to_initial), np.cos(yaw_diff_to_initial))

    if abs(yaw_diff_to_initial) > orientation_tolerance:
        print(f"   Rotation needed: {np.rad2deg(yaw_diff_to_initial):.1f}°")
        restore_success, _ = relative_move(0, 0, yaw_diff_to_initial, frame_name,
                                           robot_command_client, robot_state_client)
        if restore_success:
            print(f"✅ Initial orientation restored")
        else:
            print(f"⚠️ Failed to restore initial orientation")

    print(f"🔙 Returning to previous waypoint...")
    #recording_interface.navigate_to_previous_waypoint(steps_back=3)
    distance_covered = 0.0
    step_size = 0.5
    while distance_covered < total_distance:
            # Calculate remaining distance
            remaining = total_distance - distance_covered
            current_step = min(step_size, remaining)

            print(f"\n--- Step: {distance_covered:.2f}/{total_distance:.2f}m ---")

            # Check whether the path ahead is clear
            nav_analysis = check_path_clear(local_grid_client, robot_state_client)

            print(f"📊 Path analysis:")
            print(f"  Front: {'🟢 FREE' if not nav_analysis['front_blocked'] else '🔴 BLOCKED'} "
                  f"({nav_analysis['front_free_ratio']*100:.0f}%)")
            print(f"  Left: {'🟢 FREE' if nav_analysis['left_free'] else '🔴 BLOCKED'} "
                  f"({nav_analysis['left_free_ratio']*100:.0f}%)")
            print(f"  Right: {'🟢 FREE' if nav_analysis['right_free'] else '🔴 BLOCKED'} "
                  f"({nav_analysis['right_free_ratio']*100:.0f}%)")

            if not nav_analysis['front_blocked']:
                # Path is clear, proceed
                print(f"✅ Path clear, advancing {current_step:.2f}m")
                success = relative_move(current_step, dy, dyaw, frame_name,
                                       robot_command_client, robot_state_client)
                if success:
                    distance_covered += current_step
                else:
                    print("❌ Movement failed")
                    return False
            else:
                # Obstacle ahead: attempt avoidance
                print(f"\n🚨 OBSTACLE DETECTED! Attempting lateral avoidance...")

                # Decide avoidance direction
                if nav_analysis['recommendation'] == 'TURN_RIGHT' and nav_analysis['right_free']:
                    lateral_dir = -lateral_offset  # negative = right in body frame
                    direction_name = "RIGHT"
                elif nav_analysis['recommendation'] == 'TURN_LEFT' and nav_analysis['left_free']:
                    lateral_dir = lateral_offset  # positive = left
                    direction_name = "LEFT"
                elif nav_analysis['right_free']:
                    lateral_dir = -lateral_offset
                    direction_name = "RIGHT"
                elif nav_analysis['left_free']:
                    lateral_dir = lateral_offset
                    direction_name = "LEFT"
                else:
                    # Blocked in all directions
                    print(f"\n🔴 BLOCKED IN ALL DIRECTIONS!")
                    print(f"🔙 Returning to previous waypoint (3-4 steps back) to retry with custom obstacle avoidance...")

                    # Return to previous waypoint (3-4 steps back)
                    recording_interface.navigate_to_previous_waypoint(steps_back=3)
                    return False

                print(f"🔄 Attempting lateral avoidance to {direction_name} ({abs(lateral_dir):.1f}m)")

                # Perform lateral movement
                success_lateral = relative_move(0, lateral_dir, 0, frame_name,
                                               robot_command_client, robot_state_client)

                if not success_lateral:
                    print(f"❌ Lateral movement failed")
                    print(f"🔙 Returning to previous waypoint (3-4 steps back)...")
                    recording_interface.navigate_to_previous_waypoint(steps_back=3)
                    return False

                # Re-check path after lateral move
                nav_analysis_after = check_path_clear(local_grid_client, robot_state_client,
                                                     front_distance=current_step + 0.5)

                if not nav_analysis_after['front_blocked']:
                    print(f"✅ Path clear after lateral move to {direction_name}!")
                    # Advance
                    success = relative_move(current_step, 0, 0, frame_name,
                                           robot_command_client, robot_state_client)
                    if success:
                        distance_covered += current_step
                        # Realign: return to original track
                        print(f"🔄 Realigning to original trajectory...")
                        relative_move(0, -lateral_dir, 0, frame_name,
                                     robot_command_client, robot_state_client)
                    else:
                        print("❌ Movement failed after avoidance")
                        recording_interface.navigate_to_previous_waypoint(steps_back=3)
                        return False
                else:
                    # Still blocked after lateral move
                    print(f"❌ Still blocked after lateral move to {direction_name}")
                    print(f"🔙 Returning to previous waypoint (3-4 steps back)...")
                    recording_interface.navigate_to_previous_waypoint(steps_back=3)
                    return False

            time.sleep(0.5)  # Small delay between moves

    print(f"\n✅ MOVEMENT COMPLETE: {total_distance:.2f}m traveled successfully!")
    return True


def easy_walk(options):
    bosdyn.client.util.setup_logging(options.verbose)
    sdk = bosdyn.client.create_standard_sdk('easyWalk')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
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
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        state_client = robot.ensure_client(RobotStateClient.default_service_name)
        local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)
        robot.time_sync.wait_for_sync()
        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on()
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')
        blocking_stand(command_client)

        recordingInterface.start_recording()

        # proto = local_grid_client.get_local_grids(['no_step'])
        # pts, cells_no_step, color = create_vtk_no_step_grid(proto, robot_state_client)
        #
        # # Extract x, y coordinates (in meters in the VISION frame)
        # x = pts[:, 0]
        # y = pts[:, 1]
        #
        # # Get robot position and orientation in the VISION frame
        # robot_state = robot_state_client.get_robot_state()
        # vision_tform_body = get_a_tform_b(
        #     robot_state.kinematic_state.transforms_snapshot,
        #     VISION_FRAME_NAME,
        #     BODY_FRAME_NAME
        # )
        #
        # robot_x = vision_tform_body.position.x
        # robot_y = vision_tform_body.position.y
        # # Extract yaw from quaternion (rotation around Z axis)
        # quat = vision_tform_body.rotation
        # # Compute yaw from quaternion: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        # robot_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
        #                        1.0 - 2.0 * (quat.y**2 + quat.z**2))
        #
        # # Analyze navigation zones
        # nav_analysis = analyze_navigation_zones(pts, cells_no_step, robot_x, robot_y, robot_yaw,
        #                                        front_distance=2.0, lateral_distance=1.5, lateral_width=1.0)
        #
        #
        # # Visualize the no-step grid
        # print(f"No-step grid: {len(pts)} cells")
        # print(f"  - Non-steppable cells (no-step <= 0): {np.sum(cells_no_step <= 0.0)}")
        # print(f"  - Steppable cells (no-step > 0): {np.sum(cells_no_step > 0.0)}")
        # print(f"\nRobot position in VISION frame:")
        # print(f"  - X: {robot_x:.2f} m")
        # print(f"  - Y: {robot_y:.2f} m")
        # print(f"  - Yaw (orientation): {np.rad2deg(robot_yaw):.1f}°")
        #
        # print(f"\n=== NAVIGATION ZONE ANALYSIS ===")
        # print(f"FRONT ZONE (0-2m ahead):")
        # print(f"  - Blocked: {'YES ❌' if nav_analysis['front_blocked'] else 'NO ✓'}")
        # print(f"  - Free space: {nav_analysis['front_free_ratio']*100:.1f}%")
        # print(f"\nLEFT ZONE:")
        # print(f"  - Free: {'YES ✓' if nav_analysis['left_free'] else 'NO ❌'}")
        # print(f"  - Free space: {nav_analysis['left_free_ratio']*100:.1f}%")
        # print(f"\nRIGHT ZONE:")
        # print(f"  - Free: {'YES ✓' if nav_analysis['right_free'] else 'NO ❌'}")
        # print(f"  - Free space: {nav_analysis['right_free_ratio']*100:.1f}%")
        # print(f"\n>>> RECOMMENDATION: {nav_analysis['recommendation']} <<<")
        #
        # # Plot: points colored by cells_no_step + highlight analyzed zones
        # plt.figure(figsize=(14, 12))
        #
        # # Base grid (semi-transparent)
        # colors_norm = color.astype(np.float32) / 255.0
        # plt.scatter(x, y, c=colors_norm, s=3, alpha=0.8, label='No-step grid (base)')
        #
        # # Highlight the three zones of interest with distinct colors
        # masks = nav_analysis['masks']
        # if np.any(masks['front']):
        #     plt.scatter(x[masks['front']], y[masks['front']],
        #                c='yellow', s=20, alpha=1, edgecolors='orange', linewidths=0.5,
        #                label='Front Zone', marker='s')
        # if np.any(masks['left']):
        #     plt.scatter(x[masks['left']], y[masks['left']],
        #                c='cyan', s=20, alpha=1, edgecolors='blue', linewidths=0.5,
        #                label='Left Zone', marker='^')
        # if np.any(masks['right']):
        #     plt.scatter(x[masks['right']], y[masks['right']],
        #                c='magenta', s=20, alpha=1, edgecolors='purple', linewidths=0.5,
        #                label='Right Zone', marker='v')
        #
        # # Draw robot as a marker + heading arrow
        # plt.plot(robot_x, robot_y, 'go', markersize=18, label='Robot',
        #         markeredgecolor='black', markeredgewidth=2.5, zorder=10)
        #
        # # Arrow showing heading (1.5 m length)
        # arrow_length = 1.5
        # dx_arrow = arrow_length * np.cos(robot_yaw)
        # dy_arrow = arrow_length * np.sin(robot_yaw)
        # plt.arrow(robot_x, robot_y, dx_arrow, dy_arrow,
        #          head_width=0.4, head_length=0.3, fc='lime', ec='darkgreen', linewidth=3,
        #          label='Direction', zorder=9)
        #
        # plt.xlabel('X [m] (VISION)', fontsize=12)
        # plt.ylabel('Y [m] (VISION)', fontsize=12)
        #
        # title_color = 'green' if nav_analysis['recommendation'] == 'GO_STRAIGHT' else \
        #               'blue' if nav_analysis['recommendation'] == 'TURN_LEFT' else \
        #               'orange' if nav_analysis['recommendation'] == 'TURN_RIGHT' else 'red'
        #
        # plt.title(f'Navigation Analysis: {nav_analysis["recommendation"]}\n' +
        #          f'(Front: {nav_analysis["front_free_ratio"]*100:.0f}% free, ' +
        #          f'L: {nav_analysis["left_free_ratio"]*100:.0f}%, R: {nav_analysis["right_free_ratio"]*100:.0f}%)',
        #          fontsize=14, fontweight='bold', color=title_color)
        # plt.axis('equal')
        # plt.grid(True, alpha=0.3)
        # plt.legend(loc='upper right', fontsize=10)
        # plt.tight_layout()
        # plt.show()
        #
        # # === MISSIONE CON OBSTACLE AVOIDANCE ===
        # print(f"\n{'='*70}")
        # print(f"🚀 MISSION START: Movement with obstacle avoidance")
        # print(f"{'='*70}\n")
        #
        # # Esempio: vai avanti di 7 metri con obstacle avoidance integrato
        # mission_success = safe_relative_move(
        #     dx=10.0,           # 7 metri avanti
        #     dy=0.0,           # nessuno spostamento laterale iniziale
        #     dyaw=0.0,         # mantieni orientamento
        #     frame_name="vision",
        #     robot_command_client=command_client,
        #     robot_state_client=robot_state_client,
        #     local_grid_client=local_grid_client,
        #     recording_interface=recordingInterface,
        #     lateral_offset=1,  # spostamento laterale di 2m per evitare ostacoli
        #     max_retries=4
        # )
        recordingInterface.get_recording_status()
        recordingInterface.create_default_waypoint()
        recordingInterface.get_recording_status()
        relative_move(0, 0, math.radians(5), "vision", command_client, robot_state_client)
        recordingInterface.create_default_waypoint()
        recordingInterface.get_recording_status()
        relative_move(7, 0, 0, "vision", command_client, robot_state_client)
        recordingInterface.create_default_waypoint()
        recordingInterface.get_recording_status()
        relative_move(0, 0, - math.radians(90), "vision", command_client, robot_state_client)
        recordingInterface.create_default_waypoint()
        recordingInterface.get_recording_status()
        relative_move(2, 0, 0, "vision", command_client, robot_state_client)
        recordingInterface.create_default_waypoint()
        recordingInterface.get_recording_status()

        recordingInterface.create_new_edge()



        # if mission_success:
        #     robot.logger.info('✅ Mission completed successfully!')
        #     print(f"\n{'='*70}")
        #     print(f"✅ MISSION COMPLETED SUCCESSFULLY!")
        #     print(f"{'='*70}\n")
        # else:
        #     robot.logger.info('❌ Mission failed - robot returned to initial position')
        #     print(f"\n{'='*70}")
        #     print(f"❌ MISSION FAILED - Returned to initial position")
        #     print(f"{'='*70}\n")

        robot.logger.info('Robot mission completed.')
        log_comment = 'Easy autowalk with obstacle avoidance.'
        robot.operator_comment(log_comment)
        robot.logger.info('Added comment "%s" to robot log.', log_comment)

        # Stop recording and download the graph
        recordingInterface.stop_recording()
        recordingInterface.navigate_to_first_waypoint()
        command_client.robot_command(RobotCommandBuilder.synchro_sit_command(), end_time_secs=time.time() + 20)
        sleep(1)
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


# FIXME Change hostname for Jetson
def main():
    # Instead of argparse, create an options object manually
    options = SimpleNamespace()
    options.hostname = "192.168.50.5"
    options.verbose = False
    options.recording_user_name = ""
    options.recording_session_name = ""
    options.download_filepath = os.getcwd()

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
