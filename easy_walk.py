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


def find_nearest_waypoint_to_cell(env, target_cell, recording_interface):
    """
    Find the nearest waypoint to a target cell.

    Args:
        env: EnvironmentMap instance
        target_cell: (row, col) tuple
        recording_interface: RecordingInterface to access waypoints

    Returns:
        str: Waypoint ID of nearest waypoint, or None if no waypoints exist
    """
    # Get target world position
    target_world = env.get_world_position_from_cell(target_cell[0], target_cell[1])
    if target_world is None:
        return None

    target_x, target_y = target_world

    # Get all waypoints from the graph
    waypoints = recording_interface.get_waypoint_list()
    if not waypoints:
        print("[WARNING] No waypoints available yet")
        return None

    # Find closest waypoint
    min_distance = float('inf')
    nearest_waypoint_id = None

    for wp in waypoints:
        # Get waypoint position from graph
        wp_x = wp.waypoint_tform_ko.position.x
        wp_y = wp.waypoint_tform_ko.position.y

        # Calculate distance
        dist = np.sqrt((wp_x - target_x)**2 + (wp_y - target_y)**2)

        if dist < min_distance:
            min_distance = dist
            nearest_waypoint_id = wp.id

    print(f"[NAV] Nearest waypoint to cell {target_cell}: {nearest_waypoint_id} (distance: {min_distance:.2f}m)")
    return nearest_waypoint_id


def navigate_to_cell_via_waypoint(local_grid_client, robot_state_client, command_client,
                                   env, recording_interface, target_cell):
    """
    Navigate to a cell by first going to the nearest waypoint, then moving to the cell.

    Returns:
        tuple: (success: bool, target_cell: tuple)
    """
    print(f"\n[NAV] Navigating to cell {target_cell} via waypoint...")

    # Find nearest waypoint
    waypoint_id = find_nearest_waypoint_to_cell(env, target_cell, recording_interface)
    if waypoint_id is None:
        print("[ERROR] No waypoints available for navigation")
        return False, target_cell

    # Navigate to that waypoint using graph_nav
    print(f"[NAV] Step 1: Navigating to waypoint {waypoint_id}...")
    nav_success = recording_interface.navigate_to_waypoint(waypoint_id)

    if not nav_success:
        print(f"[ERROR] Failed to navigate to waypoint {waypoint_id}")
        return False, target_cell

    print(f"[OK] Reached waypoint {waypoint_id}")
    time.sleep(1)

    # Now move from waypoint to target cell
    print(f"[NAV] Step 2: Moving from waypoint to target cell {target_cell}...")

    # Get target world position
    world_pos = env.get_world_position_from_cell(target_cell[0], target_cell[1])
    if world_pos is None:
        print(f"[ERROR] Failed to get world position for cell {target_cell}")
        return False, target_cell

    target_x, target_y = world_pos

    # Get current robot position
    proto = local_grid_client.get_local_grids(['no_step'])
    pts, cells_no_step, color = spotGrid.create_vtk_no_step_grid(proto, robot_state_client)

    local_grid_proto = None
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'no_step':
            local_grid_proto = local_grid_found
            break

    if local_grid_proto is None:
        print("[ERROR] No 'no_step' grid found")
        return False, target_cell

    transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
    vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)

    robot_x = vision_tform_body.position.x
    robot_y = vision_tform_body.position.y

    # Calculate movement needed
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = np.sqrt(dx**2 + dy**2)

    print(f"[NAV] Distance from waypoint to cell: {distance:.2f}m")

    # Check if path is clear
    path_clear = check_line_of_sight(robot_x, robot_y, target_x, target_y, pts, cells_no_step)

    if not path_clear:
        print(f"[WARNING] Path from waypoint to cell {target_cell} is still BLOCKED")
        return False, target_cell

    # Calculate yaw
    target_yaw = np.arctan2(dy, dx)
    quat = vision_tform_body.rotation
    current_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                             1.0 - 2.0 * (quat.y**2 + quat.z**2))
    dyaw = target_yaw - current_yaw
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))

    # Rotate and move
    print(f"[NAV] Rotating {np.rad2deg(dyaw):.1f}° and moving {distance:.2f}m to cell...")

    success_rot = movements.relative_move(0, 0, dyaw, "vision", command_client, robot_state_client)
    if not success_rot:
        print("[ERROR] Failed to rotate towards cell")
        return False, target_cell

    time.sleep(0.5)

    success_move = movements.relative_move(distance, 0, 0, "vision", command_client, robot_state_client)
    if not success_move:
        print("[ERROR] Failed to move to cell")
        return False, target_cell

    print(f"[OK] Successfully reached cell {target_cell} via waypoint navigation!")
    return True, target_cell


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
                                   candidates, chosen_point, iteration, env=None):
    """
    Visualize the no-step grid with sampled candidates and chosen point.
    Optionally overlay global grid map.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot local grid points
    x = pts[:, 0]
    y = pts[:, 1]
    colors_norm = color.astype(np.float32) / 255.0
    ax.scatter(x, y, c=colors_norm, s=2, alpha=0.4, label='Local Grid (obstacles)')

    # Overlay global grid if provided
    if env is not None:
        for row in range(env.rows):
            for col in range(env.cols):
                # Get world position of cell center
                world_pos = env.get_world_position_from_cell(row, col)
                if world_pos is None:
                    continue

                cell_x, cell_y = world_pos
                half_size = env.cell_size / 2.0

                # Calculate corners in grid frame
                grid_corners = [
                    (-half_size, -half_size),
                    (half_size, -half_size),
                    (half_size, half_size),
                    (-half_size, half_size)
                ]

                # Rotate corners to world frame
                cos_yaw = np.cos(env.origin_yaw)
                sin_yaw = np.sin(env.origin_yaw)

                world_corners = []
                for gx, gy in grid_corners:
                    wx = cell_x + (gx * cos_yaw - gy * sin_yaw)
                    wy = cell_y + (gx * sin_yaw + gy * cos_yaw)
                    world_corners.append((wx, wy))

                # Draw cell
                cell_status = env.get_cell_status(row, col)
                if cell_status == 1:
                    # Visited - green fill
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkgreen',
                                          facecolor='lightgreen', alpha=0.3, zorder=2)
                else:
                    # Unvisited - gray outline only
                    rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='gray',
                                          facecolor='none', alpha=0.6, linestyle='--', zorder=2)
                ax.add_patch(rect)

                # Add cell label
                ax.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center',
                       fontsize=7, color='black', weight='bold', zorder=3,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot rejected candidates (red X)
    if 'rejected' in candidates:
        for point in candidates['rejected']:
            ax.plot(point[0], point[1], 'rx', markersize=10, markeredgewidth=2.5, zorder=5)

    # Plot valid candidates (yellow circles)
    if 'valid' in candidates:
        for point in candidates['valid']:
            ax.plot(point[0], point[1], 'yo', markersize=10, markerfacecolor='yellow',
                    markeredgewidth=2, markeredgecolor='orange', zorder=5)

    # Plot chosen point (large green star)
    if chosen_point is not None:
        ax.plot(chosen_point[0], chosen_point[1], 'g*', markersize=25,
                markeredgewidth=2, label='Target', zorder=6)

        # Draw line from robot to chosen point
        ax.plot([robot_x, chosen_point[0]], [robot_y, chosen_point[1]],
                'g--', linewidth=2.5, alpha=0.8, zorder=4)

    # Draw robot position and orientation
    ax.plot(robot_x, robot_y, 'bo', markersize=18, label='Robot', zorder=7)

    # Draw robot orientation arrow
    arrow_length = 0.3
    ax.arrow(robot_x, robot_y, arrow_length, 0,
            head_width=0.15, head_length=0.1, fc='blue', ec='blue',
            linewidth=2, zorder=7)

    # Add distance circles
    for r in [1.0, 2.0]:
        circle = patches.Circle((robot_x, robot_y), r, fill=False,
                               linestyle=':', linewidth=1,
                               edgecolor='blue', alpha=0.3, zorder=1)
        ax.add_patch(circle)

    ax.set_xlabel('X [m] (VISION)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m] (VISION)', fontsize=12, fontweight='bold')

    title = f'Iteration {iteration}: Path Planning\n'
    if env:
        visited = sum(sum(row) for row in env.map)
        total = env.rows * env.cols
        title += f'Global Grid: {visited}/{total} cells visited | '
    title += 'Yellow=valid | Red X=blocked | Green star=target'

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.pause(0.5)
    plt.close()


def move_to_next_cell_in_path(local_grid_client, robot_state_client, command_client, env,
                              path, current_path_index, recording_interface, attempted_cells=None):
    """
    Move to the next cell in the serpentine path.
    Checks if the path is clear before moving.
    If blocked, navigates to next free cell via waypoint.

    Args:
        path: List of (row, col) cells in serpentine order
        current_path_index: Current index in the path
        recording_interface: RecordingInterface for graph navigation
        attempted_cells: Set of (row, col) tuples representing cells that were attempted but failed

    Returns:
        tuple: (success: bool, next_index: int, target_cell: tuple or None)
    """
    if attempted_cells is None:
        attempted_cells = set()

    print(f"\n{'='*60}")
    print(f"[PATH FOLLOW] Path index: {current_path_index}/{len(path)}")
    print(f"[INFO] Attempted but unreachable cells: {len(attempted_cells)}")
    print(f"{'='*60}")

    # Find next unvisited cell in path (starting from current_path_index)
    target_cell = None
    next_index = current_path_index

    # Get current map state
    print(f"[MAP] Current exploration map:")
    env.print_map()

    # Search for next unvisited cell
    for i in range(current_path_index, len(path)):
        cell = path[i]
        row, col = cell

        # Check if already visited
        cell_status = env.get_cell_status(row, col)

        # Check if previously failed
        if cell in attempted_cells:
            print(f"[SKIP] Cell ({row},{col}) at index {i}: Previously unreachable")
            continue

        if cell_status == 1:
            print(f"[SKIP] Cell ({row},{col}) at index {i}: Already visited")
            continue

        # Found next target!
        target_cell = cell
        next_index = i
        print(f"[TARGET] Cell ({row},{col}) at index {i}: Next target")
        break

    if target_cell is None:
        print("[INFO] All cells in path have been visited or attempted!")
        return False, current_path_index, None

    target_row, target_col = target_cell

    # Get world position for target cell (center of cell)
    world_pos = env.get_world_position_from_cell(target_row, target_col)
    if world_pos is None:
        print(f"[ERROR] Failed to convert cell ({target_row},{target_col}) to world coordinates")
        return False, next_index + 1, target_cell

    target_x, target_y = world_pos
    print(f"[INFO] Target world position: ({target_x:.2f}, {target_y:.2f})")

    # Get local grid to check obstacles
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
        return False, next_index, target_cell

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

    # Calculate distance and direction
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = np.sqrt(dx**2 + dy**2)

    print(f"[INFO] Distance to target: {distance:.2f}m")

    # Check if path is clear
    path_clear = check_line_of_sight(robot_x, robot_y, target_x, target_y,
                                     pts, cells_no_step)

    if not path_clear:
        print(f"[WARNING] Path to cell ({target_row},{target_col}) is BLOCKED")
        print(f"[INFO] Looking for next free cell with lower index in path...")

        # Visualize the blocked path
        visualize_grid_with_candidates(
            pts, cells_no_step, color, robot_x, robot_y,
            {'rejected': [(target_x, target_y)], 'valid': []},
            None, next_index, env
        )

        # Find next free cell (not visited, not attempted, not blocked)
        alternative_cell = None
        alternative_index = None

        for i in range(next_index + 1, len(path)):
            alt_cell = path[i]
            alt_row, alt_col = alt_cell

            # Skip if already visited
            if env.get_cell_status(alt_row, alt_col) == 1:
                continue

            # Skip if already attempted
            if alt_cell in attempted_cells:
                continue

            # Check if this cell has a clear path
            alt_world_pos = env.get_world_position_from_cell(alt_row, alt_col)
            if alt_world_pos is None:
                continue

            alt_x, alt_y = alt_world_pos
            alt_path_clear = check_line_of_sight(robot_x, robot_y, alt_x, alt_y, pts, cells_no_step)

            if alt_path_clear:
                # Found a reachable alternative!
                alternative_cell = alt_cell
                alternative_index = i
                print(f"[INFO] Found alternative free cell: {alt_cell} at index {i}")
                break
            else:
                navigate_to_cell_via_waypoint(local_grid_client, robot_state_client, command_client, env, recording_interface, alt_world_pos)
                break


        if alternative_cell is None:
            print(f"[WARNING] No free cells found ahead, will mark {target_cell} as unreachable")
            return False, next_index + 1, target_cell

        # Navigate to alternative cell via waypoint
        print(f"[INFO] Using graph navigation to reach alternative cell {alternative_cell}...")

        nav_success, _ = navigate_to_cell_via_waypoint(
            local_grid_client,
            robot_state_client,
            command_client,
            env,
            recording_interface,
            alternative_cell
        )

        if nav_success:
            print(f"[OK] Successfully reached alternative cell {alternative_cell} via waypoint!")
            # Return success and update index to alternative cell
            return True, alternative_index + 1, alternative_cell
        else:
            print(f"[ERROR] Failed to reach alternative cell {alternative_cell} via waypoint")
            # Mark both original and alternative as unreachable
            return False, alternative_index + 1, alternative_cell

    print(f"[OK] Path to cell ({target_row},{target_col}) is CLEAR")

    # Visualize target
    visualize_grid_with_candidates(
        pts, cells_no_step, color, robot_x, robot_y,
        {'rejected': [], 'valid': [(target_x, target_y)]},
        (target_x, target_y), next_index, env
    )

    # Calculate yaw to face the target
    target_yaw = np.arctan2(dy, dx)

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
        return False, next_index + 1, target_cell

    time.sleep(0.5)

    # Then move forward
    print(f"[INFO] Step 2: Moving forward {distance:.2f}m...")
    success_move = movements.relative_move(distance, 0, 0, "vision",
                                          command_client, robot_state_client)

    if success_move:
        print(f"[OK] Successfully reached cell ({target_row},{target_col})!")
        return True, next_index + 1, target_cell
    else:
        print(f"[ERROR] Failed to reach cell ({target_row},{target_col})")
        return False, next_index + 1, target_cell


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

        env = environmentMap.EnvironmentMap(rows=3, cols=3, cell_size=0.5)
        x_boot, y_boot, z_boot, quat_boot = spotUtils.getPosition(robot_state_client)

        yaw_boot = np.arctan2(2.0 * (quat_boot.w * quat_boot.z + quat_boot.x * quat_boot.y),
                              1.0 - 2.0 * (quat_boot.y**2 + quat_boot.z**2))

        # Set origin with position AND orientation
        env.set_origin(x_boot, y_boot, yaw_boot, start_row=0, start_col=0)

        print(f'[INIT] Boot position: x={x_boot:.3f}, y={y_boot:.3f}, z={z_boot:.3f}')
        print(f'[INIT] Boot orientation: yaw={np.rad2deg(yaw_boot):.1f}°')

        # Generate serpentine path (lawnmower pattern)
        path = env.generate_serpentine_path()
        print(f"\n[PATH] Generated serpentine path with {len(path)} cells")
        print(f"[PATH] Pattern: ")
        for i, cell in enumerate(path[:20]):  # Show first 20 cells
            print(f"  {i+1}: {cell}")
        if len(path) > 20:
            print(f"  ... ({len(path)-20} more cells)")

        # Track cells that were attempted but couldn't be reached
        attempted_cells = set()
        cell_attempt_count = {}  # Track how many times we tried each cell
        max_attempts_per_cell = 3  # Give up on a cell after this many failures

        # Follow the serpentine path
        current_path_index = 0
        consecutive_failures = 0
        max_consecutive_failures = 5

        while current_path_index < len(path):
            print(f"\n{'#'*70}")
            print(f"### PATH STEP: {current_path_index + 1}/{len(path)} ###")
            print(f"{'#'*70}\n")

            # Move to next cell in path
            success, next_index, target_cell = move_to_next_cell_in_path(
                local_grid_client,
                robot_state_client,
                command_client,
                env,
                path,
                current_path_index,
                recordingInterface,
                attempted_cells=attempted_cells
            )

            # Update path index
            current_path_index = next_index

            if success:
                # Reset failure counter on success
                consecutive_failures = 0

                # Get new position
                x, y, z, _ = spotUtils.getPosition(robot_state_client)
                print(f'[POS] Current position: x={x:.3f}, y={y:.3f}')
                print(f'[POS] Delta from boot: dx={x-x_boot:.3f}, dy={y-y_boot:.3f}')

                # ONLY update map if movement was successful
                cell = env.update_position(x, y)
                env.print_map()

                # Remove from attempted cells if we successfully reached it
                if target_cell and target_cell in attempted_cells:
                    attempted_cells.remove(target_cell)
                    if target_cell in cell_attempt_count:
                        del cell_attempt_count[target_cell]
                    print(f"[OK] Cell {target_cell} successfully explored and marked!")

                # Create waypoint
                recordingInterface.create_default_waypoint()
                print(f"[OK] Waypoint created at cell {target_cell}")

                time.sleep(1)
            else:
                # Check if we've exhausted the path
                if target_cell is None:
                    print("[INFO] All reachable cells have been explored!")
                    break

                consecutive_failures += 1
                print(f"[WARNING] Failed to reach cell {target_cell} (consecutive failures: {consecutive_failures}/{max_consecutive_failures})")

                # Track failed attempt for this cell
                if target_cell:
                    if target_cell not in cell_attempt_count:
                        cell_attempt_count[target_cell] = 0
                    cell_attempt_count[target_cell] += 1

                    print(f"[TRACK] Cell {target_cell}: attempt {cell_attempt_count[target_cell]}/{max_attempts_per_cell}")

                    # If we've tried too many times, give up on this cell
                    if cell_attempt_count[target_cell] >= max_attempts_per_cell:
                        attempted_cells.add(target_cell)
                        print(f"[SKIP] Cell {target_cell} marked as unreachable after {max_attempts_per_cell} attempts")

        # Print final exploration statistics
        print(f"\n{'='*70}")
        print(f"EXPLORATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total cells in path: {len(path)}")
        print(f"Cells explored: {sum(sum(row) for row in env.map)}")
        print(f"Unreachable cells: {len(attempted_cells)}")
        if attempted_cells:
            print(f"Unreachable cells list: {sorted(attempted_cells)}")
        print(f"Final map state:")
        env.print_map()

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
