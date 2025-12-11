import math
import os
import sys
import time
from time import sleep
import numpy as np

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.api import local_grid_pb2
from types import SimpleNamespace
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.estop import EstopClient
from matplotlib import pyplot as plt
import navGraphUtils
import movements
import spotGrid
import spotLogInUtils


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
    pts, cells_no_step, color = spotGrid.create_vtk_no_step_grid(proto, robot_state_client)

    # Extract the local grid proto to access its transforms_snapshot
    local_grid_proto = None
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'no_step':
            local_grid_proto = local_grid_found
            break

    if local_grid_proto is None:
        print("[WARNING] No 'no_step' grid found in response")
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
        print(f"[WARNING] Failed to get robot transform from grid snapshot: {e}")
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
    nav_analysis = spotGrid.analyze_navigation_zones(pts, cells_no_step, robot_x, robot_y, robot_yaw,
                                           front_distance=front_distance,
                                           lateral_distance=1.5,
                                           lateral_width=1.0)

    return nav_analysis


# def safe_relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client,
#                        local_grid_client, recording_interface, lateral_offset=2.0, max_retries=2):
#     """
#     Move the robot with integrated obstacle avoidance.
#
#     Returns:
#         tuple: (success: bool, total_distance: float)
#     """
#     print(f"\n{'=' * 60}")
#     print(f"🎯 REQUESTED MOVE: dx={dx:.2f}m, dy={dy:.2f}m, dyaw={np.rad2deg(dyaw):.1f}°")
#     print(f"{'=' * 60}")
#
#     # Save initial position AND orientation
#     initial_state = robot_state_client.get_robot_state()
#     initial_vision_tform_body = get_a_tform_b(
#         initial_state.kinematic_state.transforms_snapshot,
#         VISION_FRAME_NAME,
#         BODY_FRAME_NAME
#     )
#     initial_x = initial_vision_tform_body.position.x
#     initial_y = initial_vision_tform_body.position.y
#
#     quat = initial_vision_tform_body.rotation
#     initial_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
#                              1.0 - 2.0 * (quat.y ** 2 + quat.z ** 2))
#
#     # Calculate target yaw (initial + requested rotation)
#     target_yaw = initial_yaw + dyaw
#     # Normalize to [-π, π]
#     target_yaw = np.arctan2(np.sin(target_yaw), np.cos(target_yaw))
#
#     print(f"[INFO] Mission start state:")
#     print(f"   Position: ({initial_x:.2f}, {initial_y:.2f}) m")
#     print(f"   Initial yaw: {np.rad2deg(initial_yaw):.1f}°")
#     print(f"   Target yaw: {np.rad2deg(target_yaw):.1f}°")
#
#     total_distance = abs(dx)
#     total_traveled = 0.0
#
#     # Try main movement
#     success, distance_traveled = relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client)
#     total_traveled += distance_traveled
#
#     # Calculate yaw error
#     current_state = robot_state_client.get_robot_state()
#     current_vision_tform_body = get_a_tform_b(
#         current_state.kinematic_state.transforms_snapshot,
#         VISION_FRAME_NAME,
#         BODY_FRAME_NAME
#     )
#     current_quat = current_vision_tform_body.rotation
#     current_yaw = np.arctan2(2.0 * (current_quat.w * current_quat.z + current_quat.x * current_quat.y),
#                              1.0 - 2.0 * (current_quat.y ** 2 + current_quat.z ** 2))
#
#     # Calculate yaw error
#     yaw_error = target_yaw - current_yaw
#     # Normalize to [-π, π]
#     yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
#
#     print(f"\n📐 Orientation check:")
#     print(f"   Current yaw: {np.rad2deg(current_yaw):.1f}°")
#     print(f"   Target yaw: {np.rad2deg(target_yaw):.1f}°")
#     print(f"   Error: {np.rad2deg(yaw_error):.1f}°")
#
#     # If orientation is wrong (> 5°), correct it even if movement succeeded
#     orientation_tolerance = np.deg2rad(5)  # 5° tolerance
#
#     if abs(yaw_error) > orientation_tolerance:
#         print(f"[WARNING] Orientation error detected ({np.rad2deg(yaw_error):.1f}°)")
#         print(f"[INFO] Correcting orientation before proceeding...")
#
#         restore_success, _ = relative_move(0, 0, yaw_error, frame_name,
#                                            robot_command_client, robot_state_client)
#         if restore_success:
#             print(f"[OK] Orientation corrected")
#             # If movement succeeded AND orientation was corrected -> SUCCESS
#             if success:
#                 print(f"[OK] MOVEMENT COMPLETE: {total_traveled:.2f}m traveled successfully!")
#                 return True, total_traveled
#         else:
#             print(f"[ERROR] Failed to restore orientation")
#             success = False  # Mark as failure if we can't restore orientation
#
#     if success:
#         print(f"[OK] MOVEMENT COMPLETE: {total_traveled:.2f}m traveled successfully!")
#         return True, total_traveled
#
#     print(f"[ERROR] Movement failed after {distance_traveled:.2f}m")
#     print(f"[INFO] Restoring initial orientation before going back...")
#
#     # Restore initial orientation (not target, but starting orientation)
#     yaw_diff_to_initial = initial_yaw - current_yaw
#     yaw_diff_to_initial = np.arctan2(np.sin(yaw_diff_to_initial), np.cos(yaw_diff_to_initial))
#
#     if abs(yaw_diff_to_initial) > orientation_tolerance:
#         print(f"   Rotation needed: {np.rad2deg(yaw_diff_to_initial):.1f}°")
#         restore_success, _ = relative_move(0, 0, yaw_diff_to_initial, frame_name,
#                                            robot_command_client, robot_state_client)
#         if restore_success:
#             print(f"[OK] Initial orientation restored")
#         else:
#             print(f"[WARNING] Failed to restore initial orientation")
#
#     print(f"[INFO] Returning to previous waypoint...")
#     #recording_interface.navigate_to_previous_waypoint(steps_back=3)
#     distance_covered = 0.0
#     step_size = 0.5
#     while distance_covered < total_distance:
#             # Calculate remaining distance
#             remaining = total_distance - distance_covered
#             current_step = min(step_size, remaining)
#
#             print(f"\n--- Step: {distance_covered:.2f}/{total_distance:.2f}m ---")
#
#             # Check whether the path ahead is clear
#             nav_analysis = check_path_clear(local_grid_client, robot_state_client)
#
#             print(f"[INFO] Path analysis:")
#             print(f"  Front: {'FREE' if not nav_analysis['front_blocked'] else 'BLOCKED'} "
#                   f"({nav_analysis['front_free_ratio']*100:.0f}%)")
#             print(f"  Left: {'FREE' if nav_analysis['left_free'] else 'BLOCKED'} "
#                   f"({nav_analysis['left_free_ratio']*100:.0f}%)")
#             print(f"  Right: {'FREE' if nav_analysis['right_free'] else 'BLOCKED'} "
#                   f"({nav_analysis['right_free_ratio']*100:.0f}%)")
#
#             if not nav_analysis['front_blocked']:
#                 # Path is clear, proceed
#                 print(f"[OK] Path clear, advancing {current_step:.2f}m")
#                 success = relative_move(current_step, dy, dyaw, frame_name,
#                                        robot_command_client, robot_state_client)
#                 if success:
#                     distance_covered += current_step
#                 else:
#                     print("[ERROR] Movement failed")
#                     return False
#             else:
#                 # Obstacle ahead: attempt avoidance
#                 print(f"[WARNING] OBSTACLE DETECTED! Attempting lateral avoidance...")
#
#                 # Decide avoidance direction
#                 if nav_analysis['recommendation'] == 'TURN_RIGHT' and nav_analysis['right_free']:
#                     lateral_dir = -lateral_offset  # negative = right in body frame
#                     direction_name = "RIGHT"
#                 elif nav_analysis['recommendation'] == 'TURN_LEFT' and nav_analysis['left_free']:
#                     lateral_dir = lateral_offset  # positive = left
#                     direction_name = "LEFT"
#                 elif nav_analysis['right_free']:
#                     lateral_dir = -lateral_offset
#                     direction_name = "RIGHT"
#                 elif nav_analysis['left_free']:
#                     lateral_dir = lateral_offset
#                     direction_name = "LEFT"
#                 else:
#                     # Blocked in all directions
#                     print(f"[ERROR] BLOCKED IN ALL DIRECTIONS!")
#                     print(f"[INFO] Returning to previous waypoint (3-4 steps back) to retry with custom obstacle avoidance...")
#
#                     # Return to previous waypoint (3-4 steps back)
#                     recording_interface.navigate_to_previous_waypoint(steps_back=3)
#                     return False
#
#                 print(f"[INFO] Attempting lateral avoidance to {direction_name} ({abs(lateral_dir):.1f}m)")
#
#                 # Perform lateral movement
#                 success_lateral = relative_move(0, lateral_dir, 0, frame_name,
#                                                robot_command_client, robot_state_client)
#
#                 if not success_lateral:
#                     print(f"[ERROR] Lateral movement failed")
#                     print(f"[INFO] Returning to previous waypoint (3-4 steps back)...")
#                     recording_interface.navigate_to_previous_waypoint(steps_back=3)
#                     return False
#
#                 # Re-check path after lateral move
#                 nav_analysis_after = check_path_clear(local_grid_client, robot_state_client,
#                                                      front_distance=current_step + 0.5)
#
#                 if not nav_analysis_after['front_blocked']:
#                     print(f"[OK] Path clear after lateral move to {direction_name}!")
#                     # Advance
#                     success = relative_move(current_step, 0, 0, frame_name,
#                                            robot_command_client, robot_state_client)
#                     if success:
#                         distance_covered += current_step
#                         # Realign: return to original track
#                         print(f"[INFO] Realigning to original trajectory...")
#                         relative_move(0, -lateral_dir, 0, frame_name,
#                                      robot_command_client, robot_state_client)
#                     else:
#                         print("[ERROR] Movement failed after avoidance")
#                         recording_interface.navigate_to_previous_waypoint(steps_back=3)
#                         return False
#                 else:
#                     # Still blocked after lateral move
#                     print(f"[ERROR] Still blocked after lateral move to {direction_name}")
#                     print(f"[INFO] Returning to previous waypoint (3-4 steps back)...")
#                     recording_interface.navigate_to_previous_waypoint(steps_back=3)
#                     return False
#
#             time.sleep(0.5)  # Small delay between moves
#
#     print(f"\n[OK] MOVEMENT COMPLETE: {total_distance:.2f}m traveled successfully!")
#     return True


def easy_walk(options):
    robot, lease_client, robot_state_client, client_metadata = spotLogInUtils.setLogInfo(options)

    estop = spotLogInUtils.SimpleEstop(robot, options.name + "_estop")

    recordingInterface = navGraphUtils.RecordingInterface(robot, options.download_filepath, client_metadata)
    recordingInterface.stop_recording()
    recordingInterface.clear_map()


    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)
        robot.time_sync.wait_for_sync()
        robot.logger.info('Powering on robot... This may take several seconds.')
        robot.power_on()
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')
        blocking_stand(command_client)

        recordingInterface.start_recording()

        proto = local_grid_client.get_local_grids(['no_step'])
        pts, cells_no_step, color = spotGrid.create_vtk_no_step_grid(proto, robot_state_client)

        # Extract x, y coordinates (in meters in the VISION frame)
        x = pts[:, 0]
        y = pts[:, 1]

        # Get robot position and orientationNo non lo  in the VISION frame
        robot_state = robot_state_client.get_robot_state()
        vision_tform_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            VISION_FRAME_NAME,
            BODY_FRAME_NAME
        )

        robot_x = vision_tform_body.position.x
        robot_y = vision_tform_body.position.y
        # Extract yaw from quaternion (rotation around Z axis)
        quat = vision_tform_body.rotation
        # Compute yaw from quaternion: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        robot_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                               1.0 - 2.0 * (quat.y**2 + quat.z**2))
        # Analyze navigation zones
        nav_analysis = spotGrid.analyze_navigation_zones(pts, cells_no_step, robot_x, robot_y, robot_yaw,
                                               front_distance=2.0, lateral_distance=1.5, lateral_width=1.0,
                                               rear_distance=1.0)


        # Visualize the no-step grid
        print(f"No-step grid: {len(pts)} cells")
        print(f"  - Non-steppable cells (no-step <= 0): {np.sum(cells_no_step <= 0.0)}")
        print(f"  - Steppable cells (no-step > 0): {np.sum(cells_no_step > 0.0)}")
        print(f"\nRobot position in VISION frame:")
        print(f"  - X: {robot_x:.2f} m")
        print(f"  - Y: {robot_y:.2f} m")
        print(f"  - Yaw (orientation): {np.rad2deg(robot_yaw):.1f}°")

        print(f"\n=== NAVIGATION ZONE ANALYSIS ===")
        print(f"FRONT ZONE (0-2m ahead):")
        print(f"  - Blocked: {'YES' if nav_analysis['front_blocked'] else 'NO'}")
        print(f"  - Free space: {nav_analysis['front_free_ratio']*100:.1f}%")
        print(f"\nLEFT ZONE:")
        print(f"  - Free: {'YES' if nav_analysis['left_free'] else 'NO'}")
        print(f"  - Free space: {nav_analysis['left_free_ratio']*100:.1f}%")
        print(f"\nRIGHT ZONE:")
        print(f"  - Free: {'YES' if nav_analysis['right_free'] else 'NO'}")
        print(f"  - Free space: {nav_analysis['right_free_ratio']*100:.1f}%")
        print(f"\nREAR ZONE:")
        print(f"  - Free: {'YES' if nav_analysis['rear_free'] else 'NO'}")
        print(f"  - Free space: {nav_analysis['rear_free_ratio']*100:.1f}%")
        print(f"\n>>> RECOMMENDATION: {nav_analysis['recommendation']} <<<")

        # Plot: points colored by cells_no_step + highlight analyzed zones
        plt.figure(figsize=(14, 12))

        # Base grid (semi-transparent)
        colors_norm = color.astype(np.float32) / 255.0
        plt.scatter(x, y, c=colors_norm, s=3, alpha=0.8, label='No-step grid (base)')

        # Highlight the four zones of interest with distinct colors
        masks = nav_analysis['masks']
        if np.any(masks['front']):
            plt.scatter(x[masks['front']], y[masks['front']],
                       c='yellow', s=20, alpha=0.4, edgecolors='orange', linewidths=0.5,
                       label='Front Zone', marker='s')
        if np.any(masks['left']):
            plt.scatter(x[masks['left']], y[masks['left']],
                       c='cyan', s=20, alpha=0.4, edgecolors='blue', linewidths=0.5,
                       label='Left Zone', marker='^')
        if np.any(masks['right']):
            plt.scatter(x[masks['right']], y[masks['right']],
                       c='magenta', s=20, alpha=0.4, edgecolors='purple', linewidths=0.5,
                       label='Right Zone', marker='v')
        if np.any(masks['rear']):
            plt.scatter(x[masks['rear']], y[masks['rear']],
                       c='orange', s=20, alpha=0.4, edgecolors='red', linewidths=0.5,
                       label='Rear Zone', marker='o')

        # Draw robot as a black rectangle (Spot dimensions: ~1.1m long x 0.5m wide)
        import matplotlib.patches as patches
        from matplotlib.transforms import Affine2D

        robot_length = 1.1  # meters (front to back)
        robot_width = 0.5   # meters (left to right)

        # Create a rectangle centered at origin, then rotate and translate
        robot_rect = patches.Rectangle(
            (-robot_length/2, -robot_width/2),  # bottom-left corner when centered at origin
            robot_length, robot_width,
            fill=True,
            facecolor='black',
            edgecolor='white',
            linewidth=2,
            zorder=11
        )

        # Apply rotation (yaw) and translation (robot position)
        transform = Affine2D().rotate(robot_yaw).translate(robot_x, robot_y) + plt.gca().transData
        robot_rect.set_transform(transform)
        plt.gca().add_patch(robot_rect)

        # Draw robot center point (green dot on top of the rectangle)
        plt.plot(robot_x, robot_y, 'go', markersize=8, label='Robot Center',
                markeredgecolor='white', markeredgewidth=1.5, zorder=12)

        # Add distance reference circles around the robot (e.g., 1m, 2m)
        for r in [1.0, 2.0]:
            circle = patches.Circle((robot_x, robot_y), r,
                                    fill=False,
                                    linestyle='--',
                                    linewidth=1.0,
                                    edgecolor='gray',
                                    alpha=0.5)
            plt.gca().add_patch(circle)
            # Annotate radius
            plt.text(robot_x + r, robot_y, f"{r:.0f} m",
                     color='gray', fontsize=8, ha='left', va='bottom')

        # Arrow showing heading (1.5 m length)
        arrow_length = 1.5
        dx_arrow = arrow_length * np.cos(robot_yaw)
        dy_arrow = arrow_length * np.sin(robot_yaw)
        plt.arrow(robot_x, robot_y, dx_arrow, dy_arrow,
                 head_width=0.4, head_length=0.3, fc='lime', ec='darkgreen', linewidth=3,
                 label='Direction', zorder=9)

        plt.xlabel('X [m] (VISION)', fontsize=12)
        plt.ylabel('Y [m] (VISION)', fontsize=12)

        title_color = 'green' if nav_analysis['recommendation'] == 'GO_STRAIGHT' else \
                      'blue' if nav_analysis['recommendation'] == 'TURN_LEFT' else \
                      'orange' if nav_analysis['recommendation'] == 'TURN_RIGHT' else 'red'

        plt.title(f'Navigation Analysis: {nav_analysis["recommendation"]}\n' +
                 f'(Front: {nav_analysis["front_free_ratio"]*100:.0f}% free, ' +
                 f'L: {nav_analysis["left_free_ratio"]*100:.0f}%, R: {nav_analysis["right_free_ratio"]*100:.0f}%)',
                 fontsize=14, fontweight='bold', color=title_color)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()

        # === MISSIONE CON OBSTACLE AVOIDANCE ===
        print(f"\n{'='*70}")
        print(f"[INFO] MISSION START: Movement with obstacle avoidance")
        print(f"{'='*70}\n")

        # Esempio: vai avanti di 7 metri con obstacle avoidance integrato
        # mission_success = safe_relative_move(
        #     dx=0.0,           # 7 metri avanti
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

        # --- SIMPLE MISSION WITHOUT OBSTACLE AVOIDANCE ---

        recordingInterface.get_recording_status()
        recordingInterface.create_default_waypoint()
        recordingInterface.get_recording_status()
        movements.relative_move(0, 0, math.radians(0), "vision", command_client, robot_state_client)
        # recordingInterface.create_default_waypoint()
        # recordingInterface.get_recording_status()
        # relative_move(0, 0, 0, "vision", command_client, robot_state_client)
        # recordingInterface.create_default_waypoint()
        # recordingInterface.get_recording_status()
        # relative_move(0, 0, - math.radians(0), "vision", command_client, robot_state_client)
        # recordingInterface.create_default_waypoint()
        # recordingInterface.get_recording_status()
        # relative_move(0, 0, 0, "vision", command_client, robot_state_client)
        # recordingInterface.create_default_waypoint()
        # recordingInterface.get_recording_status()

        # --- END OF SIMPLE MISSION ---

        #recordingInterface.create_new_edge()

        robot.logger.info('Robot mission completed.')
        log_comment = 'Easy autowalk with obstacle avoidance.'
        robot.operator_comment(log_comment)
        robot.logger.info('Added comment "%s" to robot log.', log_comment)

        # Stop recording and download the graph
        recordingInterface.stop_recording()
        recordingInterface.navigate_to_first_waypoint()
        command_client.robot_command(RobotCommandBuilder.synchro_sit_command(), end_time_secs=time.time() + 20)
        sleep(5)
        recordingInterface.download_full_graph()
        estop.stop()



# FIXME Change hostname for Jetson/localhost
def main():
    # Instead of argparse, create an options object manually
    options = SimpleNamespace()
    options.name = "easyWalk"
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
