import math
import os
import sys
import time
from time import sleep, altzone
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


def sample_cell_points(env, cell_row, cell_col, num_samples=20):
    """
    Sample random points within a cell.

    Args:
        env: EnvironmentMap instance
        cell_row: Row index of the cell
        cell_col: Column index of the cell
        num_samples: Number of random points to generate

    Returns:
        list of (x, y) tuples representing sampled points in world coordinates
    """
    # Get cell center in world coordinates
    world_pos = env.get_world_position_from_cell(cell_row, cell_col)
    if world_pos is None:
        return []

    cell_center_x, cell_center_y = world_pos
    half_size = env.cell_size / 2.0

    # Generate random offsets within the cell (in grid frame)
    samples = []
    for _ in range(num_samples):
        # Random offset from center in grid frame
        offset_x = np.random.uniform(-half_size * 0.8, half_size * 0.8)  # 80% to avoid edges
        offset_y = np.random.uniform(-half_size * 0.8, half_size * 0.8)

        # Rotate offset to world frame
        cos_yaw = np.cos(env.origin_yaw)
        sin_yaw = np.sin(env.origin_yaw)

        world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
        world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw

        # Final world position
        sample_x = cell_center_x + world_offset_x
        sample_y = cell_center_y + world_offset_y

        samples.append((sample_x, sample_y))

    return samples


def find_best_point_in_cell(robot_x, robot_y, env, cell_row, cell_col, pts, cells_no_step):
    """
    Sample 20 random points in a cell and find the one with clear path that is farthest from robot.

    Args:
        robot_x, robot_y: Current robot position
        env: EnvironmentMap instance
        cell_row, cell_col: Target cell coordinates
        pts: Grid points array from local grid
        cells_no_step: Cell values from local grid

    Returns:
        tuple: (best_x, best_y, valid_samples, rejected_samples) or (None, None, [], []) if no valid point found
    """
    # Sample random points in the cell
    sampled_points = sample_cell_points(env, cell_row, cell_col, num_samples=20)

    if not sampled_points:
        return None, None, [], []

    valid_samples = []
    rejected_samples = []

    # Check each sampled point
    for sample_x, sample_y in sampled_points:
        # Check if path is clear
        if check_line_of_sight(robot_x, robot_y, sample_x, sample_y, pts, cells_no_step):
            valid_samples.append((sample_x, sample_y))
        else:
            rejected_samples.append((sample_x, sample_y))

    # If no valid samples, return None
    if not valid_samples:
        print(f"[WARNING] No clear path found to any sampled point in cell ({cell_row},{cell_col})")
        return None, None, valid_samples, rejected_samples

    # Choose the valid point that is FARTHEST from the robot
    best_point = None
    max_distance = -1

    for sample_x, sample_y in valid_samples:
        dist = np.sqrt((sample_x - robot_x)**2 + (sample_y - robot_y)**2)
        if dist > max_distance:
            max_distance = dist
            best_point = (sample_x, sample_y)

    print(f"[OK] Found {len(valid_samples)} valid points in cell ({cell_row},{cell_col}), chose farthest at {max_distance:.2f}m")

    return best_point[0], best_point[1], valid_samples, rejected_samples


def visualize_grid_with_candidates(pts, cells_no_step, color, robot_x, robot_y,
                                   candidates, chosen_point, iteration, env=None):
    """
    Visualize the no-step grid with sampled candidates and chosen point.
    Optionally overlay global grid map (only cells visible within local grid bounds).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot local grid points
    x = pts[:, 0]
    y = pts[:, 1]
    colors_norm = color.astype(np.float32) / 255.0
    ax.scatter(x, y, c=colors_norm, s=2, alpha=0.4, label='Local Grid (obstacles)')

    # Calculate local grid bounds
    local_x_min, local_x_max = x.min(), x.max()
    local_y_min, local_y_max = y.min(), y.max()

    # Overlay global grid if provided (only cells within local grid bounds)
    if env is not None:
        for row in range(env.rows):
            for col in range(env.cols):
                # Get world position of cell center
                world_pos = env.get_world_position_from_cell(row, col)
                if world_pos is None:
                    continue

                cell_x, cell_y = world_pos

                # Check if cell is within local grid bounds (with small margin)
                margin = env.cell_size
                if not (local_x_min - margin <= cell_x <= local_x_max + margin and
                       local_y_min - margin <= cell_y <= local_y_max + margin):
                    continue  # Skip cells outside local grid view

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

    # Draw robot position
    ax.plot(robot_x, robot_y, 'bo', markersize=18, label='Robot', zorder=7)

    # Add distance circles
    for r in [1.0, 2.0]:
        circle = patches.Circle((robot_x, robot_y), r, fill=False,
                               linestyle=':', linewidth=1,
                               edgecolor='blue', alpha=0.3, zorder=1)
        ax.add_patch(circle)

    # Set axis limits to focus on local grid
    ax.set_xlim(local_x_min - 0.5, local_x_max + 0.5)
    ax.set_ylim(local_y_min - 0.5, local_y_max + 0.5)

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


    # Sample 20 random points in the cell and find the best one with clear path
    print(f"[INFO] Sampling 20 random points in cell ({target_row},{target_col})...")
    target_x, target_y, valid_samples, rejected_samples = find_best_point_in_cell(
        robot_x, robot_y, env, target_row, target_col, pts, cells_no_step
    )

    if target_x is None or target_y is None:
        print(f"[WARNING] No clear path to cell ({target_row},{target_col})")
        print(f"[INFO] Looking for next free cell with lower index in path...")

        # Visualize the blocked path
        visualize_grid_with_candidates(
            pts, cells_no_step, color, robot_x, robot_y,
            {'rejected': rejected_samples, 'valid': []},
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

            # Try to find a valid point in this alternative cell
            alt_x, alt_y, alt_valid, alt_rejected = find_best_point_in_cell(
                robot_x, robot_y, env, alt_row, alt_col, pts, cells_no_step
            )

            if alt_x is not None and alt_y is not None:
                # Found a reachable alternative!
                alternative_cell = alt_cell
                print(f"[INFO] Found alternative free cell: {alt_cell} at index {i}")
                target_x = alt_x
                target_y = alt_y
                target_row = alt_row
                target_col = alt_col
                valid_samples = alt_valid
                rejected_samples = alt_rejected
                next_index = i
                break
            else:
                return False, next_index + 1, alt_cell

        if alternative_cell is None:
            print(f"[WARNING] No free cells found ahead, will mark {target_cell} as unreachable")
            return False, next_index + 1, target_cell

    print(f"[OK] Target point in cell ({target_row},{target_col}): ({target_x:.2f}, {target_y:.2f})")

    # Calculate distance and direction
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = np.sqrt(dx ** 2 + dy ** 2)

    print(f"[INFO] Distance to target: {distance:.2f}m")

    # Visualize target with sampled points
    visualize_grid_with_candidates(
        pts, cells_no_step, color, robot_x, robot_y,
        {'rejected': rejected_samples, 'valid': valid_samples},
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

        env = environmentMap.EnvironmentMap(rows=3, cols=3, cell_size=1)
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
        # Follow the serpentine path
        current_path_index = 0

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
                recordingInterface.get_recording_status()
                recordingInterface.create_default_waypoint()
                print(f"[OK] Waypoint created at cell {target_cell}")

                time.sleep(1)
            else:
                # Check if we've exhausted the path
                if target_cell is None:
                    print("[INFO] All reachable cells have been explored!")
                    break

                if target_cell is not None:
                    recordingInterface.stop_recording()
                    navigate_to_cell_via_waypoint(local_grid_client, robot_state_client, command_client, env, recordingInterface, target_cell)
                    recordingInterface.start_recording()

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
        recordingInterface.create_new_edge()

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
