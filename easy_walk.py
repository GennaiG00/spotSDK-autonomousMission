import argparse
import os
import sys
import time
import math
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client import math_helpers
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.api import local_grid_pb2
from types import SimpleNamespace


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
                            front_distance=2.0, lateral_distance=1.5, lateral_width=1.0):
    """
    Analizza le zone davanti, destra e sinistra del robot per determinare se il percorso è libero.

    Args:
        pts: array (N, 3) con coordinate [x, y, z] delle celle nel frame VISION
        cells_no_step: array (N,) con valori no-step (<=0 = non calpestabile, >0 = calpestabile)
        robot_x, robot_y: posizione del robot in metri (frame VISION)
        robot_yaw: orientamento del robot in radianti
        front_distance: quanto guardare avanti (metri)
        lateral_distance: quanto guardare lateralmente in profondità (metri)
        lateral_width: larghezza della zona laterale da considerare (metri)

    Returns:
        dict con chiavi:
            - 'front_blocked': bool, True se davanti è bloccato
            - 'left_free': bool, True se a sinistra è libero
            - 'right_free': bool, True se a destra è libero
            - 'front_free_ratio': float 0-1, percentuale celle libere davanti
            - 'left_free_ratio': float 0-1, percentuale celle libere a sinistra
            - 'right_free_ratio': float 0-1, percentuale celle libere a destra
            - 'recommendation': str, suggerimento ('GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'BLOCKED')
    """

    # Vettori direzione del robot (asse X del body frame nel frame VISION)
    front_dir_x = np.cos(robot_yaw)
    front_dir_y = np.sin(robot_yaw)

    # Vettori perpendicolari (destra e sinistra)
    right_dir_x = np.cos(robot_yaw - np.pi/2)  # -90° = destra
    right_dir_y = np.sin(robot_yaw - np.pi/2)
    left_dir_x = np.cos(robot_yaw + np.pi/2)   # +90° = sinistra
    left_dir_y = np.sin(robot_yaw + np.pi/2)

    # Coordinate relative delle celle rispetto al robot
    dx = pts[:, 0] - robot_x
    dy = pts[:, 1] - robot_y

    # Proiezione sulla direzione frontale (distanza longitudinale)
    proj_front = dx * front_dir_x + dy * front_dir_y
    # Proiezione sulla direzione laterale (distanza trasversale, + = destra, - = sinistra)
    proj_lateral = dx * right_dir_x + dy * right_dir_y

    # --- ZONA FRONTALE ---
    # Celle davanti al robot: proj_front > 0 e < front_distance, |proj_lateral| < 0.5m (larghezza robot ~0.8m)
    mask_front = (proj_front > 0) & (proj_front <= front_distance) & (np.abs(proj_lateral) <= 0.5)
    cells_front = cells_no_step[mask_front]

    if len(cells_front) > 0:
        front_free_count = np.sum(cells_front > 0.0)
        front_free_ratio = front_free_count / len(cells_front)
    else:
        front_free_ratio = 1.0  # nessuna cella = consideriamo libero

    # Soglia: se meno del 70% è libero, consideriamo bloccato
    front_blocked = front_free_ratio < 0.7

    # --- ZONA SINISTRA ---
    # Celle a sinistra: proj_lateral < 0 (sinistra), proj_front tra 0 e lateral_distance, |proj_lateral| < lateral_width
    mask_left = (proj_front > 0) & (proj_front <= lateral_distance) & \
                (proj_lateral < 0) & (proj_lateral >= -lateral_width)
    cells_left = cells_no_step[mask_left]

    if len(cells_left) > 0:
        left_free_count = np.sum(cells_left > 0.0)
        left_free_ratio = left_free_count / len(cells_left)
    else:
        left_free_ratio = 1.0

    left_free = left_free_ratio > 0.7

    # --- ZONA DESTRA ---
    # Celle a destra: proj_lateral > 0 (destra), proj_front tra 0 e lateral_distance, proj_lateral < lateral_width
    mask_right = (proj_front > 0) & (proj_front <= lateral_distance) & \
                 (proj_lateral > 0) & (proj_lateral <= lateral_width)
    cells_right = cells_no_step[mask_right]

    if len(cells_right) > 0:
        right_free_count = np.sum(cells_right > 0.0)
        right_free_ratio = right_free_count / len(cells_right)
    else:
        right_free_ratio = 1.0

    right_free = right_free_ratio > 0.7

    # --- RACCOMANDAZIONE ---
    if not front_blocked:
        recommendation = 'GO_STRAIGHT'
    elif left_free and right_free:
        # Entrambi liberi: scegli quello con più spazio
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
        'masks': {  # per debug/visualizzazione
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
        first_waypoint = None
        for waypoint in graph.waypoints:
            if waypoint.annotations.name == "waypoint_0":
                first_waypoint = waypoint
        if first_waypoint is None:
            print('No waypoint_0 found in the graph.')
            return

        nav_to_cmd_id = None
        is_finished = False
        while not is_finished:
            nav_to_cmd_id = self._graph_nav_client.navigate_to(first_waypoint.id, 1, command_id=nav_to_cmd_id)
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            is_finished = self._check_success(nav_to_cmd_id)

def relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client, stairs=False):
    """Muove il robot relativamente alla sua posizione attuale."""
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    # We do not want to command this goal in body frame because the body will move, thus shifting
    # our goal. Instead, we transform this offset to get the goal position in the output frame
    # (which will be either odom or vision).
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal
    # Command the robot to go to the goal point in the specified frame. The command will stop at the
    # new position.
    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
    end_time = 6000.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)
    # Wait until the robot has reached the goal.
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print('Failed to reach the goal')
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print('Arrived at the goal.')
            return True
        time.sleep(1)

    return True


def check_path_clear(local_grid_client, robot_state_client, front_distance=0.2, threshold=0.7):
    """
    Controlla se il percorso davanti al robot è libero.

    Returns:
        dict con 'front_blocked', 'left_free', 'right_free', 'recommendation'
    """
    proto = local_grid_client.get_local_grids(['no_step'])
    pts, cells_no_step, color = create_vtk_no_step_grid(proto, robot_state_client)

    # Ottieni posizione e orientamento del robot
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

    # Analizza le zone
    nav_analysis = analyze_navigation_zones(pts, cells_no_step, robot_x, robot_y, robot_yaw,
                                           front_distance=front_distance,
                                           lateral_distance=1.5,
                                           lateral_width=1.0)

    return nav_analysis


def safe_relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client,
                       local_grid_client, recording_interface, lateral_offset=2.0, max_retries=2):
    """
    Muove il robot con obstacle avoidance integrato.

    Se trova un ostacolo davanti:
    1. Prova a spostarsi lateralmente (destra o sinistra)
    2. Controlla se la strada è libera
    3. Se trova spazio, si sposta lateralmente e riprende il movimento originale
    4. Se bloccato ovunque, torna alla posizione iniziale con navigate_to()

    Args:
        dx, dy, dyaw: movimento desiderato
        frame_name: frame di riferimento
        robot_command_client: client comandi robot
        robot_state_client: client stato robot
        local_grid_client: client griglia locale
        recording_interface: interfaccia per navigate_to
        lateral_offset: distanza laterale per evitare ostacoli (metri)
        max_retries: numero massimo di tentativi laterali

    Returns:
        bool: True se movimento completato, False se fallito
    """

    print(f"\n{'='*60}")
    print(f"🎯 MOVIMENTO RICHIESTO: dx={dx:.2f}m, dy={dy:.2f}m, dyaw={np.rad2deg(dyaw):.1f}°")
    print(f"{'='*60}")

    # Salva posizione iniziale totale (per eventuale ritorno)
    initial_state = robot_state_client.get_robot_state()
    initial_vision_tform_body = get_a_tform_b(
        initial_state.kinematic_state.transforms_snapshot,
        VISION_FRAME_NAME,
        BODY_FRAME_NAME
    )
    initial_x = initial_vision_tform_body.position.x
    initial_y = initial_vision_tform_body.position.y

    # Distanza totale da percorrere
    total_distance = abs(dx)
    distance_covered = 0.0
    step_size = 1.0  # Muovi 1 metro alla volta per controllare ostacoli

    while distance_covered < total_distance:
        # Calcola quanto manca
        remaining = total_distance - distance_covered
        current_step = min(step_size, remaining)

        print(f"\n--- Step: {distance_covered:.2f}/{total_distance:.2f}m ---")

        # Controlla se la strada davanti è libera
        nav_analysis = check_path_clear(local_grid_client, robot_state_client)

        print(f"📊 Analisi percorso:")
        print(f"  Frontale: {'🟢 LIBERO' if not nav_analysis['front_blocked'] else '🔴 BLOCCATO'} "
              f"({nav_analysis['front_free_ratio']*100:.0f}%)")
        print(f"  Sinistra: {'🟢 LIBERO' if nav_analysis['left_free'] else '🔴 BLOCCATO'} "
              f"({nav_analysis['left_free_ratio']*100:.0f}%)")
        print(f"  Destra: {'🟢 LIBERO' if nav_analysis['right_free'] else '🔴 BLOCCATO'} "
              f"({nav_analysis['right_free_ratio']*100:.0f}%)")

        if not nav_analysis['front_blocked']:
            # Strada libera, procedi
            print(f"✅ Percorso libero, avanzo di {current_step:.2f}m")
            success = relative_move(current_step, dy, dyaw, frame_name,
                                   robot_command_client, robot_state_client)
            if success:
                distance_covered += current_step
            else:
                print("❌ Movimento fallito")
                return False
        else:
            # Ostacolo davanti! Prova manovra evasiva
            print(f"\n🚨 OSTACOLO RILEVATO! Tentativo di aggiramento...")

            # Decidi direzione evasione
            if nav_analysis['recommendation'] == 'TURN_RIGHT' and nav_analysis['right_free']:
                lateral_dir = -lateral_offset  # negativo = destra (in body frame: y negativo = destra)
                direction_name = "DESTRA"
            elif nav_analysis['recommendation'] == 'TURN_LEFT' and nav_analysis['left_free']:
                lateral_dir = lateral_offset  # positivo = sinistra
                direction_name = "SINISTRA"
            elif nav_analysis['right_free']:
                lateral_dir = -lateral_offset
                direction_name = "DESTRA"
            elif nav_analysis['left_free']:
                lateral_dir = lateral_offset
                direction_name = "SINISTRA"
            else:
                # Bloccato ovunque!
                print(f"\n🔴 BLOCCATO IN TUTTE LE DIREZIONI!")
                print(f"🔙 Ritorno alla posizione iniziale tramite navigate_to()...")

                # Torna alla posizione iniziale
                recording_interface.navigate_to()
                return False

            print(f"🔄 Tentativo aggiramento a {direction_name} ({abs(lateral_dir):.1f}m)")

            # Spostamento laterale
            success_lateral = relative_move(0, lateral_dir, 0, frame_name,
                                           robot_command_client, robot_state_client)

            if not success_lateral:
                print(f"❌ Spostamento laterale fallito")
                print(f"🔙 Ritorno alla posizione iniziale...")
                recording_interface.navigate_to()
                return False

            # Controlla se ora la strada è libera
            nav_analysis_after = check_path_clear(local_grid_client, robot_state_client,
                                                 front_distance=current_step + 0.5)

            if not nav_analysis_after['front_blocked']:
                print(f"✅ Strada libera dopo spostamento a {direction_name}!")
                # Avanza
                success = relative_move(current_step, 0, 0, frame_name,
                                       robot_command_client, robot_state_client)
                if success:
                    distance_covered += current_step
                    # Riallineamento: torna sulla traiettoria originale
                    print(f"🔄 Riallineamento sulla traiettoria originale...")
                    relative_move(0, -lateral_dir, 0, frame_name,
                                 robot_command_client, robot_state_client)
                else:
                    print("❌ Movimento fallito dopo aggiramento")
                    recording_interface.navigate_to()
                    return False
            else:
                # Ancora bloccato dopo spostamento laterale
                print(f"❌ Ancora bloccato dopo spostamento a {direction_name}")
                print(f"🔙 Ritorno alla posizione iniziale...")
                recording_interface.navigate_to()
                return False

        time.sleep(0.5)  # Piccola pausa tra i movimenti

    print(f"\n✅ MOVIMENTO COMPLETATO: {total_distance:.2f}m percorsi con successo!")
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

        proto = local_grid_client.get_local_grids(['no_step'])
        pts, cells_no_step, color = create_vtk_no_step_grid(proto, robot_state_client)

        # Estrai coordinate x, y (in metri nel frame VISION)
        x = pts[:, 0]
        y = pts[:, 1]

        # Ottieni posizione e orientamento del robot nel frame VISION
        robot_state = robot_state_client.get_robot_state()
        vision_tform_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            VISION_FRAME_NAME,
            BODY_FRAME_NAME
        )

        robot_x = vision_tform_body.position.x
        robot_y = vision_tform_body.position.y
        # Estrai lo yaw dal quaternione (rotazione attorno all'asse Z)
        quat = vision_tform_body.rotation
        # Calcolo dello yaw da quaternione: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        robot_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                               1.0 - 2.0 * (quat.y**2 + quat.z**2))

        # Analizza le zone di navigazione
        nav_analysis = analyze_navigation_zones(pts, cells_no_step, robot_x, robot_y, robot_yaw,
                                               front_distance=2.0, lateral_distance=1.5, lateral_width=1.0)


        # Visualizza la griglia no-step
        print(f"No-step grid: {len(pts)} celle")
        print(f"  - Celle non-calpestabili (no-step <= 0): {np.sum(cells_no_step <= 0.0)}")
        print(f"  - Celle calpestabili (no-step > 0): {np.sum(cells_no_step > 0.0)}")
        print(f"\nPosizione robot nel frame VISION:")
        print(f"  - X: {robot_x:.2f} m")
        print(f"  - Y: {robot_y:.2f} m")
        print(f"  - Yaw (orientamento): {np.rad2deg(robot_yaw):.1f}°")

        print(f"\n=== ANALISI ZONE DI NAVIGAZIONE ===")
        print(f"Zona FRONTALE (0-2m davanti):")
        print(f"  - Bloccata: {'SÌ ❌' if nav_analysis['front_blocked'] else 'NO ✓'}")
        print(f"  - Spazio libero: {nav_analysis['front_free_ratio']*100:.1f}%")
        print(f"\nZona SINISTRA:")
        print(f"  - Libera: {'SÌ ✓' if nav_analysis['left_free'] else 'NO ❌'}")
        print(f"  - Spazio libero: {nav_analysis['left_free_ratio']*100:.1f}%")
        print(f"\nZona DESTRA:")
        print(f"  - Libera: {'SÌ ✓' if nav_analysis['right_free'] else 'NO ❌'}")
        print(f"  - Spazio libero: {nav_analysis['right_free_ratio']*100:.1f}%")
        print(f"\n>>> RACCOMANDAZIONE: {nav_analysis['recommendation']} <<<")

        # Mappa raccomandazioni a descrizioni
        rec_map = {
            'GO_STRAIGHT': '🟢 Vai dritto, percorso libero',
            'TURN_LEFT': '🔵 Svolta a SINISTRA, percorso laterale libero',
            'TURN_RIGHT': '🟠 Svolta a DESTRA, percorso laterale libero',
            'BLOCKED': '🔴 BLOCCATO: ostacolo insuperabile davanti e ai lati'
        }
        print(f"    {rec_map.get(nav_analysis['recommendation'], nav_analysis['recommendation'])}")

        # Plot: punti colorati in base a cells_no_step + evidenzia zone analizzate
        plt.figure(figsize=(14, 12))

        # Griglia di base (semi-trasparente)
        colors_norm = color.astype(np.float32) / 255.0
        plt.scatter(x, y, c=colors_norm, s=3, alpha=0.3, label='No-step grid (base)')

        # Evidenzia le tre zone di interesse con colori distinti
        masks = nav_analysis['masks']
        if np.any(masks['front']):
            plt.scatter(x[masks['front']], y[masks['front']],
                       c='yellow', s=20, alpha=0.8, edgecolors='orange', linewidths=0.5,
                       label='Zona FRONTALE', marker='s')
        if np.any(masks['left']):
            plt.scatter(x[masks['left']], y[masks['left']],
                       c='cyan', s=20, alpha=0.8, edgecolors='blue', linewidths=0.5,
                       label='Zona SINISTRA', marker='^')
        if np.any(masks['right']):
            plt.scatter(x[masks['right']], y[masks['right']],
                       c='magenta', s=20, alpha=0.8, edgecolors='purple', linewidths=0.5,
                       label='Zona DESTRA', marker='v')

        # Disegna il robot come marker + freccia per orientamento
        plt.plot(robot_x, robot_y, 'go', markersize=18, label='Robot',
                markeredgecolor='black', markeredgewidth=2.5, zorder=10)

        # Freccia che indica la direzione di marcia (lunghezza 1.5 metri)
        arrow_length = 1.5
        dx_arrow = arrow_length * np.cos(robot_yaw)
        dy_arrow = arrow_length * np.sin(robot_yaw)
        plt.arrow(robot_x, robot_y, dx_arrow, dy_arrow,
                 head_width=0.4, head_length=0.3, fc='lime', ec='darkgreen', linewidth=3,
                 label='Direzione', zorder=9)

        plt.xlabel('X [m] (VISION)', fontsize=12)
        plt.ylabel('Y [m] (VISION)', fontsize=12)

        title_color = 'green' if nav_analysis['recommendation'] == 'GO_STRAIGHT' else \
                     'blue' if nav_analysis['recommendation'] == 'TURN_LEFT' else \
                     'orange' if nav_analysis['recommendation'] == 'TURN_RIGHT' else 'red'

        plt.title(f'Analisi Navigazione: {nav_analysis["recommendation"]}\n' +
                 f'(Frontale: {nav_analysis["front_free_ratio"]*100:.0f}% libero, ' +
                 f'Sx: {nav_analysis["left_free_ratio"]*100:.0f}%, Dx: {nav_analysis["right_free_ratio"]*100:.0f}%)',
                 fontsize=14, fontweight='bold', color=title_color)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()

        # === MISSIONE CON OBSTACLE AVOIDANCE ===
        print(f"\n{'='*70}")
        print(f"🚀 INIZIO MISSIONE: Movimento di 7 metri con obstacle avoidance")
        print(f"{'='*70}\n")

        # Esempio: vai avanti di 7 metri con obstacle avoidance integrato
        mission_success = safe_relative_move(
            dx=5.0,           # 7 metri avanti
            dy=0.0,           # nessuno spostamento laterale iniziale
            dyaw=0.0,         # mantieni orientamento
            frame_name="vision",
            robot_command_client=command_client,
            robot_state_client=robot_state_client,
            local_grid_client=local_grid_client,
            recording_interface=recordingInterface,
            lateral_offset=0.5,  # spostamento laterale di 2m per evitare ostacoli
            max_retries=2
        )

        if mission_success:
            robot.logger.info('✅ Missione completata con successo!')
            print(f"\n{'='*70}")
            print(f"✅ MISSIONE COMPLETATA CON SUCCESSO!")
            print(f"{'='*70}\n")
        else:
            robot.logger.info('❌ Missione fallita - robot tornato alla posizione iniziale')
            print(f"\n{'='*70}")
            print(f"❌ MISSIONE FALLITA - Tornato alla posizione iniziale")
            print(f"{'='*70}\n")

        robot.logger.info('Robot mission completed.')
        log_comment = 'Easy autowalk with obstacle avoidance.'
        robot.operator_comment(log_comment)
        robot.logger.info('Added comment "%s" to robot log.', log_comment)

        # Ferma recording e scarica grafo
        recordingInterface.stop_recording()
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


#TODO drifting problem when come back to first waypoint
def main():
    # Invece di usare argparse, creiamo un oggetto options a mano
    options = SimpleNamespace()
    options.hostname = "192.168.80.3"
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
