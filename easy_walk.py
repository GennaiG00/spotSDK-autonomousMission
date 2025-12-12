import math
import os
import sys
import time
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.frame_helpers import get_a_tform_b
from types import SimpleNamespace
import navGraphUtils
import movements
import spotGrid
import spotLogInUtils
import environmentMap
import spotUtils

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


def check_line_of_sight(x1, y1, x2, y2, pts, cells, obstacle_threshold=0.0):
    """
    Check if there's a clear line of sight between two points.
    Uses sampling along the line to check for obstacles.

    Args:
        x1, y1: Start coordinates (robot position)
        x2, y2: End coordinates (target point)
        pts: Grid points array
        cells: Grid cell values (<=0=obstacle, >0=free)
        obstacle_threshold: Cell value below which it's considered blocked

    Returns:
        bool: True if path is clear, False if blocked
    """
    # Number of points to check along the line
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    num_checks = max(10, int(distance * 10))  # 10 checks per meter

    for i in range(num_checks):
        t = i / max(1, num_checks - 1)
        check_x = x1 + t * (x2 - x1)
        check_y = y1 + t * (y2 - y1)

        # Find nearest grid point
        distances = np.sqrt((pts[:, 0] - check_x)**2 + (pts[:, 1] - check_y)**2)
        nearest_idx = np.argmin(distances)

        # Check if this point is an obstacle (cells_no_step <= 0 means obstacle)
        if cells[nearest_idx] <= obstacle_threshold:
            return False  # Path blocked

    return True  # Path clear


def visualize_grid_with_candidates(pts, cells_no_step, color, robot_x, robot_y,
                                   candidates, chosen_point, iteration):
    """
    Visualize the no-step grid with sampled candidates and chosen point.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.figure(figsize=(12, 10))

    # Plot grid points
    x = pts[:, 0]
    y = pts[:, 1]
    colors_norm = color.astype(np.float32) / 255.0
    plt.scatter(x, y, c=colors_norm, s=2, alpha=0.6, label='Grid')

    # Plot rejected candidates (red X)
    if 'rejected' in candidates:
        for point in candidates['rejected']:
            plt.plot(point[0], point[1], 'rx', markersize=8, markeredgewidth=2)

    # Plot valid candidates (green circles)
    if 'valid' in candidates:
        for point in candidates['valid']:
            plt.plot(point[0], point[1], 'go', markersize=8, markerfacecolor='none',
                    markeredgewidth=2)

    # Plot chosen point (large green star)
    if chosen_point is not None:
        plt.plot(chosen_point[0], chosen_point[1], 'g*', markersize=20,
                markeredgewidth=2, label='Chosen point')

        # Draw line from robot to chosen point
        plt.plot([robot_x, chosen_point[0]], [robot_y, chosen_point[1]],
                'g--', linewidth=2, alpha=0.7)

    # Draw robot position (large blue dot)
    plt.plot(robot_x, robot_y, 'bo', markersize=15, label='Robot')

    # Add distance circles
    for r in [1.0, 2.0, 3.0]:
        circle = patches.Circle((robot_x, robot_y), r, fill=False,
                               linestyle='--', linewidth=1,
                               edgecolor='blue', alpha=0.3)
        plt.gca().add_patch(circle)

    plt.xlabel('X [m] (VISION)', fontsize=11)
    plt.ylabel('Y [m] (VISION)', fontsize=11)
    plt.title(f'Iteration {iteration}: Random Sampling\n'
              f'Green circles=valid | Red X=blocked | Green star=chosen',
              fontsize=12, fontweight='bold')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def sample_and_move_to_free_point(local_grid_client, robot_state_client, command_client,
                                  num_samples=20, iteration=1):
    """
    Sample random points in the no-step grid, check if path is clear,
    and move to the best reachable point.

    Returns:
        tuple: (success: bool, chosen_point: tuple or None)
    """
    print(f"\n{'='*60}")
    print(f"[SAMPLING] Iteration {iteration}: Sampling {num_samples} random points")
    print(f"{'='*60}")

    # Get local grid
    proto = local_grid_client.get_local_grids(['no_step'])
    pts, cells_no_step, color = spotGrid.create_vtk_no_step_grid(proto, robot_state_client)

    # Get grid proto
    local_grid_proto = None
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'no_step':
            local_grid_proto = local_grid_found
            break

    if local_grid_proto is None:
        print("[ERROR] No 'no_step' grid found")
        return False, None

    transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot

    # Get robot position
    vision_tform_body = get_a_tform_b(
        transforms_snapshot,
        VISION_FRAME_NAME,
        BODY_FRAME_NAME
    )
    robot_x = vision_tform_body.position.x
    robot_y = vision_tform_body.position.y

    print(f"[INFO] Robot position: ({robot_x:.2f}, {robot_y:.2f})")
    print(f"[INFO] Grid has {len(pts)} points")

    # Sample random points
    valid_candidates = []
    rejected_candidates = []

    for i in range(num_samples):
        # Pick random point from grid
        random_idx = np.random.randint(0, len(pts))
        target_point = pts[random_idx]
        target_x, target_y = target_point[0], target_point[1]

        # Calculate distance and direction
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = np.sqrt(dx**2 + dy**2)

        # Skip points too close (<1.0m) or too far (>4m)
        if distance < 1.0 or distance > 4.0:
            rejected_candidates.append((target_x, target_y))
            continue

        # Check if target point itself is free
        if cells_no_step[random_idx] <= 0:
            print(f"[X] Point {i+1}: ({target_x:.2f}, {target_y:.2f}) - Target is obstacle")
            rejected_candidates.append((target_x, target_y))
            continue

        # Check if path is clear
        path_clear = check_line_of_sight(robot_x, robot_y, target_x, target_y,
                                         pts, cells_no_step)

        if path_clear:
            valid_candidates.append({
                'point': (target_x, target_y),
                'distance': distance,
                'dx': dx,
                'dy': dy,
                'cell_value': cells_no_step[random_idx]
            })
            print(f"[OK] Point {i+1}: ({target_x:.2f}, {target_y:.2f}) "
                  f"dist={distance:.2f}m - PATH CLEAR")
        else:
            print(f"[X] Point {i+1}: ({target_x:.2f}, {target_y:.2f}) "
                  f"dist={distance:.2f}m - BLOCKED")
            rejected_candidates.append((target_x, target_y))

    if not valid_candidates:
        print("[ERROR] No valid reachable points found!")

        # Visualize anyway to show why nothing was found
        visualize_grid_with_candidates(
            pts, cells_no_step, color, robot_x, robot_y,
            {'rejected': rejected_candidates, 'valid': []},
            None, iteration
        )

        # UNSTUCK BEHAVIOR: If no valid points, try to back up and rotate
        print("[UNSTUCK] Attempting to get unstuck...")
        print("[UNSTUCK] Step 1: Backing up 1.5m")
        backup_success = movements.relative_move(-1, 0, 0, "vision",
                                                command_client, robot_state_client)

        if backup_success:
            print("[UNSTUCK] Step 2: Rotating 90° to explore new direction")
            rotate_success = movements.relative_move(0, 0, np.pi/2, "vision",
                                                    command_client, robot_state_client)
            if rotate_success:
                print("[OK] Unstuck maneuver completed, will retry next iteration")
                return False, None

        print("[WARNING] Unstuck maneuver failed")
        return False, None

    # Sort by distance (prefer FARTHEST points for better exploration)
    valid_candidates.sort(key=lambda x: x['distance'], reverse=True)

    # Choose best point (FARTHEST with clear path)
    best = valid_candidates[0]
    chosen_point = best['point']

    print(f"\n[INFO] Choosing point: ({chosen_point[0]:.2f}, {chosen_point[1]:.2f})")
    print(f"       Distance: {best['distance']:.2f}m")
    print(f"       Valid candidates found: {len(valid_candidates)}")

    # Visualize grid with candidates
    valid_points = [c['point'] for c in valid_candidates]
    visualize_grid_with_candidates(
        pts, cells_no_step, color, robot_x, robot_y,
        {'rejected': rejected_candidates, 'valid': valid_points},
        chosen_point, iteration
    )

    # Calculate yaw to face the target
    target_yaw = np.arctan2(best['dy'], best['dx'])

    # Get current yaw
    quat = vision_tform_body.rotation
    current_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                             1.0 - 2.0 * (quat.y**2 + quat.z**2))

    # Calculate rotation needed
    dyaw = target_yaw - current_yaw
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))  # Normalize to [-π, π]

    print(f"[INFO] Required rotation: {np.rad2deg(dyaw):.1f}°")

    # First rotate to face target
    print("[INFO] Step 1: Rotating to face target...")
    success_rot = movements.relative_move(0, 0, dyaw, "vision",
                                         command_client, robot_state_client)

    if not success_rot:
        print("[ERROR] Failed to rotate")
        return False, chosen_point

    time.sleep(0.5)

    # Then move forward
    print(f"[INFO] Step 2: Moving forward {best['distance']:.2f}m...")
    success_move = movements.relative_move(best['distance'], 0, 0, "vision",
                                          command_client, robot_state_client)

    if success_move:
        print(f"[OK] Successfully reached target point!")
        return True, chosen_point
    else:
        print(f"[ERROR] Failed to reach target point")
        return False, chosen_point


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
        resp = recordingInterface.create_default_waypoint()

        env = environmentMap.EnvironmentMap(cell_size=1, rows=10, cols=10)
        x_boot, y_boot, z_boot = spotUtils.getPosition(robot_state_client)
        env.set_origin(x_boot, y_boot, start_row=5, start_col=5)  # Start from center of grid

        print(f'[INIT] Boot position: x={x_boot:.3f}, y={y_boot:.3f}, z={z_boot:.3f}')

        # Random exploration loop
        num_iterations = 10
        consecutive_failures = 0
        max_consecutive_failures = 3

        for iteration in range(1, num_iterations + 1):
            print(f"\n{'#'*70}")
            print(f"### ITERATION {iteration}/{num_iterations} ###")
            print(f"{'#'*70}\n")

            # Sample and move to random free point
            success, chosen_point = sample_and_move_to_free_point(
                local_grid_client,
                robot_state_client,
                command_client,
                num_samples=20,
                iteration=iteration
            )

            if success:
                # Reset failure counter on success
                consecutive_failures = 0

                # Get new position
                x, y, z = spotUtils.getPosition(robot_state_client)
                print(f'[POS] Current position: x={x:.3f}, y={y:.3f}')
                print(f'[POS] Delta from boot: dx={x-x_boot:.3f}, dy={y-y_boot:.3f}')

                # Update map
                env.update_position(x, y)
                env.print_map()

                # Create waypoint
                recordingInterface.create_default_waypoint()
                print(f"[OK] Waypoint created for iteration {iteration}")

                time.sleep(1)
            else:
                consecutive_failures += 1
                print(f"[WARNING] Iteration {iteration} failed (consecutive failures: {consecutive_failures}/{max_consecutive_failures})")

                # If too many consecutive failures, try more aggressive unstuck
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n[UNSTUCK] Too many failures! Attempting aggressive recovery...")

                    # Rotate 180° to face completely opposite direction
                    print("[UNSTUCK] Rotating 180° to explore opposite direction")
                    movements.relative_move(0, 0, np.pi, "vision",
                                          command_client, robot_state_client)
                    time.sleep(1)

                    # Reset counter after aggressive recovery
                    consecutive_failures = 0

                time.sleep(0.5)

        recordingInterface.get_recording_status()
        recordingInterface.create_default_waypoint()
        recordingInterface.get_recording_status()

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
        robot.power_off(cut_immediately=False)
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
