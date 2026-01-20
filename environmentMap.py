import numpy as np

class EnvironmentMap(object):
    def __init__(self, rows=12, cols=12, cell_size=2.0):
        self.map = [[0 for _ in range(cols)] for _ in range(rows)]
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size  # Size of each cell in meters
        self.origin_x = 0.0  # World X coordinate of grid origin
        self.origin_y = 0.0  # World Y coordinate of grid origin
        self.origin_yaw = 0.0  # Initial yaw orientation of robot (radians)
        self.start_cell = (0, 0)  # Starting cell in grid coordinates
        self.current_cell = (0, 0)
        # Track explored sides for each cell: key=(row,col), value=4-bit binary (NESW)
        # Bit 3 (0b1000): North, Bit 2 (0b0100): East, Bit 1 (0b0010): South, Bit 0 (0b0001): West
        self.explored_sides = {}
        # Track waypoint positions: list of (x, y) world coordinates
        self.waypoints = []
        # Track robot path: list of tuples (x, y, movement_type) where movement_type is:
        # 'explore' = exploring new cells, 'navigate' = navigating on graph to reposition
        self.robot_path = []
        # Map values:
        # 0 = unvisited/unknown
        # 1 = visited and accessible
        # -1 = attempted but blocked (obstacle detected)

    def mark_explored_side(self, path=None, target_index=None, robot_row=None, robot_col=None,
                           target_row=None, target_col=None):
        """
        Mark which side of target cell was explored based on robot's approach direction.

        Recommended usage:
        - Use path + target_index for the TARGET cell (from serpentine path)
        - Use robot_row/robot_col for ROBOT position (actual position from sensors)

        Call patterns:
        1. Path + target_index + robot position (recommended):
           mark_explored_side(path=path, target_index=5, robot_row=1, robot_col=2)

        2. Explicit coordinates only (backward compatible):
           mark_explored_side(target_row=2, target_col=3, robot_row=1, robot_col=3)

        Bit encoding:
            North (↑) = 0b1000 (bit 3) = 8
            East  (→) = 0b0100 (bit 2) = 4
            South (↓) = 0b0010 (bit 1) = 2
            West  (←) = 0b0001 (bit 0) = 1

        The function automatically determines which side based on geometry:
        - If robot is ABOVE target (lower row) → marks NORTH side
        - If robot is BELOW target (higher row) → marks SOUTH side
        - If robot is LEFT of target (lower col) → marks WEST side
        - If robot is RIGHT of target (higher col) → marks EAST side

        Args:
            path: List of (row, col) tuples representing the serpentine path (optional)
            target_index: Index in path of the target cell (optional)
            robot_row: Row of the robot's current position (required)
            robot_col: Column of the robot's current position (required)
            target_row: Row of the target cell (optional, alternative to path/target_index)
            target_col: Column of the target cell (optional, alternative to path/target_index)

        Returns:
            int: Updated sides value for the cell (4-bit binary)

        Example:
            # Using path + target_index + robot position (recommended)
            mark_explored_side(path=path, target_index=5, robot_row=1, robot_col=2)

            # Using explicit coordinates only
            mark_explored_side(target_row=2, target_col=3, robot_row=2, robot_col=2)
        """
        # Extract target coordinates from path if provided
        if path is not None and target_index is not None:
            if 0 <= target_index < len(path):
                target_row, target_col = path[target_index]
                print(f"[PATH] Using target from path index {target_index}: ({target_row},{target_col})")
            else:
                # print(f"[ERROR] Invalid target index: {target_index}, path_len={len(path)}")
                return 0b0000

        # Validate that we have all necessary coordinates
        if target_row is None or target_col is None or robot_row is None or robot_col is None:
            print(f"[ERROR] Missing coordinates: target=({target_row},{target_col}), robot=({robot_row},{robot_col})")
            return 0b0000

        print(f"[MARK] Target: ({target_row},{target_col}), Robot: ({robot_row},{robot_col})")

        cell_key = (target_row, target_col)

        # Initialize if not exists
        if cell_key not in self.explored_sides:
            self.explored_sides[cell_key] = 0b0000

        # Calculate position difference
        delta_row = target_row - robot_row  # Positive = robot is above, Negative = robot is below
        delta_col = target_col - robot_col  # Positive = robot is left, Negative = robot is right

        # Determine the primary direction (handle diagonal by choosing stronger component)
        if abs(delta_row) > abs(delta_col):
            # VERTICAL movement dominates
            if delta_row <= 0:
                # delta_row > 0 → target_row > robot_row → Robot is ABOVE (North of) target
                # We're trying to enter from the NORTH side

                side = 0b1000  # North (bit 3)
                side_name = "North (↑)"
            else:
                # delta_row < 0 → target_row < robot_row → Robot is BELOW (South of) target
                # We're trying to enter from the SOUTH side
                side = 0b0010  # South (bit 1)
                side_name = "South (↓)"
        else:
            # HORIZONTAL movement dominates or equal
            if delta_col <= 0:
                # delta_col > 0 → target_col > robot_col → Robot is LEFT (West of) target
                # We're trying to enter from the WEST side
                side = 0b0001  # West (bit 0)
                side_name = "West (←)"
            else:
                # delta_col < 0 → target_col < robot_col → Robot is RIGHT (East of) target
                # We're trying to enter from the EAST side
                side = 0b0100  # East (bit 2)
                side_name = "East (→)"

        # Mark this side as explored using bitwise OR
        self.explored_sides[cell_key] |= side

        print(f"[SIDES] Cell ({target_row},{target_col}) - Marked {side_name} from ({robot_row},{robot_col}): {bin(side)} -> Total: {bin(self.explored_sides[cell_key])}")

        return self.explored_sides[cell_key]

    def mark_cell_fully_explored(self, row, col):
        """
        Mark a cell as fully explored (all 4 sides: 0b1111).

        Args:
            row: Row of the cell
            col: Column of the cell
        """
        cell_key = (row, col)
        self.explored_sides[cell_key] = 0b1111
        print(f"[SIDES] Cell ({row},{col}) marked as FULLY explored (0b1111)")

    def mark_cell_blocked(self, row, col):
        """
        Mark a cell as blocked (obstacle detected, cannot enter).
        Sets the map value to -1 to indicate this cell was attempted but is blocked.

        Args:
            row: Row of the cell
            col: Column of the cell
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.map[row][col] = -1
            print(f"[BLOCKED] Cell ({row},{col}) marked as blocked (value=-1)")
        else:
            print(f"[ERROR] Cannot mark cell ({row},{col}) as blocked - out of bounds")

    def is_cell_blocked(self, row, col):
        """
        Check if a cell is marked as blocked.

        Args:
            row: Row of the cell
            col: Column of the cell

        Returns:
            bool: True if cell is blocked (value=-1), False otherwise
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.map[row][col] == -1
        return False

    def return_visited_cells_near_blocked(self, path=None, blocked_index=None, robot_row=None, robot_col=None,
                                          blocked_row=None, blocked_col=None):
        """
        Return visited cells adjacent to a blocked cell relative to robot's movement direction.

        Recommended usage:
        - Use path + blocked_index for BLOCKED cell (from serpentine path)
        - Use robot_row/robot_col for ROBOT position (to determine movement direction)

        Directions are relative to robot's movement in serpentine pattern:
        - North (avanti): direction robot is moving toward
        - South (indietro): direction robot came from
        - East (destra): right side relative to movement
        - West (sinistra): left side relative to movement

        Args:
            path: List of (row, col) tuples representing the serpentine path (optional)
            blocked_index: Index in path of the blocked cell (optional)
            robot_row: Current robot row (to determine movement direction)
            robot_col: Current robot column (to determine movement direction)
            blocked_row: Row of the blocked cell (optional, alternative to path/blocked_index)
            blocked_col: Column of the blocked cell (optional, alternative to path/blocked_index)

        Returns:
            dict: Dictionary with keys 'north', 'south', 'east', 'west'.
                  Each value is either (row, col) tuple if that neighbor is visited,
                  or None if not visited or out of bounds.

        Example:
            # Using path + blocked_index + robot position (recommended)
            return_visited_cells_near_blocked(path=path, blocked_index=5, robot_row=1, robot_col=2)

            # Using explicit coordinates
            return_visited_cells_near_blocked(blocked_row=1, blocked_col=1, robot_row=1, robot_col=2)
        """
        # Extract blocked cell coordinates from path if provided
        if path is not None and blocked_index is not None:
            if 0 <= blocked_index < len(path):
                blocked_row, blocked_col = path[blocked_index]
                print(f"[PATH] Using blocked cell from path index {blocked_index}: ({blocked_row},{blocked_col})")
            else:
                # print(f"[ERROR] Invalid blocked index: {blocked_index}, path_len={len(path)}")
                return {'north': None, 'south': None, 'east': None, 'west': None}

        # Validate that we have blocked cell coordinates
        if blocked_row is None or blocked_col is None:
            print(f"[ERROR] Missing blocked cell coordinates: ({blocked_row},{blocked_col})")
            return {'north': None, 'south': None, 'east': None, 'west': None}
        visited_neighbors = {
            'north': None,  # Avanti
            'south': None,  # Indietro
            'east': None,   # Destra
            'west': None    # Sinistra
        }

        # If robot position not provided, use absolute grid directions (backward compatibility)
        if robot_row is None or robot_col is None:
            print("[WARNING] Robot position not provided, using absolute grid directions")
            directions = [
                (-1, 0, 'north'),  # North: row - 1
                (1, 0, 'south'),   # South: row + 1
                (0, 1, 'east'),    # East: col + 1
                (0, -1, 'west')    # West: col - 1
            ]
        else:
            # Determine movement direction based on robot position relative to blocked cell
            delta_row = blocked_row - robot_row
            delta_col = blocked_col - robot_col

            # Determine primary movement direction
            if abs(delta_col) > abs(delta_row):
                # Horizontal movement (left/right in serpentine)
                if delta_col > 0:
                    # Robot is LEFT of target, moving RIGHT (→)
                    # North=avanti(→), South=indietro(←), East=destra(↓), West=sinistra(↑)
                    directions = [
                        (0, 1, 'north'),    # North (avanti): col + 1
                        (0, -1, 'south'),   # South (indietro): col - 1
                        (1, 0, 'east'),     # East (destra): row + 1
                        (-1, 0, 'west')     # West (sinistra): row - 1
                    ]
                    print(f"[DIRECTION] Moving RIGHT (→): North=→, South=←, East=↓, West=↑")
                else:
                    # Robot is RIGHT of target, moving LEFT (←)
                    # North=avanti(←), South=indietro(→), East=destra(↓), West=sinistra(↑)
                    directions = [
                        (0, -1, 'north'),   # North (avanti): col - 1
                        (0, 1, 'south'),    # South (indietro): col + 1
                        (1, 0, 'east'),     # East (destra): row + 1
                        (-1, 0, 'west')     # West (sinistra): row - 1
                    ]
                    print(f"[DIRECTION] Moving LEFT (←): North=←, South=→, East=↓, West=↑")
            else:
                # Vertical movement (up/down in serpentine)
                if delta_row > 0:
                    # Robot is ABOVE target, moving DOWN (↓)
                    # North=avanti(↓), South=indietro(↑), East=destra(→), West=sinistra(←)
                    directions = [
                        (1, 0, 'north'),    # North (avanti): row + 1
                        (-1, 0, 'south'),   # South (indietro): row - 1
                        (0, 1, 'east'),     # East (destra): col + 1
                        (0, -1, 'west')     # West (sinistra): col - 1
                    ]
                    print(f"[DIRECTION] Moving DOWN (↓): North=↓, South=↑, East=→, West=←")
                else:
                    # Robot is BELOW target, moving UP (↑)
                    # North=avanti(↑), South=indietro(↓), East=destra(→), West=sinistra(←)
                    directions = [
                        (-1, 0, 'north'),   # North (avanti): row - 1
                        (1, 0, 'south'),    # South (indietro): row + 1
                        (0, 1, 'east'),     # East (destra): col + 1
                        (0, -1, 'west')     # West (sinistra): col - 1
                    ]
                    print(f"[DIRECTION] Moving UP (↑): North=↑, South=↓, East=→, West=←")

        for dr, dc, direction in directions:
            neighbor_row = blocked_row + dr
            neighbor_col = blocked_col + dc

            # Check if neighbor is within bounds
            if 0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols:
                # Check if neighbor is visited
                if self.map[neighbor_row][neighbor_col] == 1 and self.map[neighbor_row][neighbor_col] != self.map[robot_row][neighbor_col]:
                    visited_neighbors[direction] = (neighbor_row, neighbor_col)

        # Print summary
        visited_count = sum(1 for v in visited_neighbors.values() if v is not None)
        print(f"[NEIGHBORS] Cell ({blocked_row},{blocked_col}) has {visited_count}/4 visited neighbors:")
        for direction, cell in visited_neighbors.items():
            if cell:
                print(f"  {direction.capitalize()}: {cell} ✓")
            else:
                print(f"  {direction.capitalize()}: Not visited or out of bounds")

        return visited_neighbors

    def get_visited_neighbors_list(self, path=None, blocked_index=None, robot_row=None, robot_col=None,
                                    blocked_row=None, blocked_col=None):
        """
        Return a simple list of visited neighbor cells (without None values).

        Args:
            path: List of (row, col) tuples representing the serpentine path (optional)
            blocked_index: Index in path of the blocked cell (optional)
            robot_row: Current robot row (optional, for directional context)
            robot_col: Current robot column (optional, for directional context)
            blocked_row: Row of the blocked cell (optional, alternative)
            blocked_col: Column of the blocked cell (optional, alternative)

        Returns:
            list: List of (row, col) tuples of visited adjacent cells
        """
        neighbors_dict = self.return_visited_cells_near_blocked(
            path=path, blocked_index=blocked_index,
            robot_row=robot_row, robot_col=robot_col,
            blocked_row=blocked_row, blocked_col=blocked_col
        )
        return [cell for cell in neighbors_dict.values() if cell is not None]

    def get_blocked_neighbors_with_unexplored_side(self, cell_row, cell_col, path=None, path_index=None):
        """
        Find adjacent cells that are marked as blocked (-1) but have the side facing
        the current cell unexplored.

        This method checks all 4 neighbors (North, South, East, West) of the input cell and returns those that:
        1. Are marked as blocked (map[row][col] == -1)
        2. Have at least one side explored (sides != 0b0000) - means we attempted to enter before
        3. Have the side facing the input cell NOT yet explored (the side that connects to current cell)

        The side correspondence is:
        - North neighbor (row-1, col): check its SOUTH side (0b0010)
        - South neighbor (row+1, col): check its NORTH side (0b1000)
        - East neighbor (row, col+1): check its WEST side (0b0001)
        - West neighbor (row, col-1): check its EAST side (0b0100)

        Args:
            cell_row: Row of the current cell (where robot is now)
            cell_col: Column of the current cell (where robot is now)
            path: List of (row, col) tuples representing the serpentine path (optional, for logging)
            path_index: Current index in the path (optional, for logging)

        Returns:
            list or None: List of (row, col) tuples of blocked neighbors with unexplored sides
                         facing the current cell, or None if no such neighbors exist.

        Example:
            # If robot is at (1, 2) and cell (1, 3) to the East is blocked with sides=0b1010
            # (North and South explored but not East/West), this method will return [(1, 3)]
            # because the West side of (1,3) facing (1,2) is unexplored (0b0001 not set).
        """
        blocked_neighbors = []

        print(f"\n[CHECK] Looking for blocked neighbors with unexplored sides around cell ({cell_row},{cell_col})")

        # Define neighbors in absolute grid directions
        # Format: (delta_row, delta_col, direction_name, side_bit_to_check)
        neighbors = [
            (-1, 0, 'North', 0b0010),  # North neighbor: check its South side (facing us)
            (1, 0, 'South', 0b1000),   # South neighbor: check its North side (facing us)
            (0, 1, 'East', 0b0001),    # East neighbor: check its West side (facing us)
            (0, -1, 'West', 0b0100)    # West neighbor: check its East side (facing us)
        ]

        for dr, dc, direction, side_bit in neighbors:
            neighbor_row = cell_row + dr
            neighbor_col = cell_col + dc

            # Check if neighbor is within bounds
            if not (0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols):
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Out of bounds")
                continue

            # Check if neighbor is marked as blocked (-1)
            if self.map[neighbor_row][neighbor_col] != -1:
                status = "visited" if self.map[neighbor_row][neighbor_col] == 1 else "unvisited"
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Not blocked (status={status})")
                continue

            # Get explored sides of the blocked neighbor
            neighbor_sides = self.get_cell_sides_status(neighbor_row, neighbor_col)

            # Check if at least one side was explored (means we attempted before)
            if neighbor_sides == 0b0000:
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Blocked but never attempted (sides=0b0000)")
                continue

            # Check if the side facing the current cell is NOT explored
            if not (neighbor_sides & side_bit):
                # This side is NOT explored yet!
                blocked_neighbors.append((neighbor_row, neighbor_col))
                print(f"  {direction} ({neighbor_row},{neighbor_col}): ✓ FOUND! Blocked with sides={bin(neighbor_sides)}, "
                      f"side {bin(side_bit)} facing current cell is UNEXPLORED")
            else:
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Blocked but side {bin(side_bit)} "
                      f"facing current cell already explored (sides={bin(neighbor_sides)})")

        if blocked_neighbors:
            #print(f"[RESULT] Found {len(blocked_neighbors)} blocked neighbor(s) with unexplored sides: {blocked_neighbors}\n")
            return blocked_neighbors
        else:
            print(f"[RESULT] No blocked neighbors with unexplored sides found for cell ({cell_row},{cell_col})\n")
            return None

    def get_unknow_neighbors_with_unexplored_side(self, cell_row, cell_col, path=None, path_index=None):
        """
        Find adjacent cells that are UNTESTED (value=0) and appear BEFORE the current cell in the serpentine path.

        This method checks all 4 neighbors (North, South, East, West) of the input cell and returns those that:
        1. Are untested (map[row][col] == 0) - never attempted sampling
        2. Have a path index LOWER than the current cell (appear before in serpentine)
        3. Have the side facing the input cell NOT yet explored

        Args:
            cell_row: Row of the current cell (where robot is now)
            cell_col: Column of the current cell (where robot is now)
            path: List of (row, col) tuples representing the serpentine path (required)
            path_index: Current index in the path (required)

        Returns:
            list or None: List of (row, col) tuples of untested neighbors with lower path index,
                         or None if no such neighbors exist.

        Example:
            # If robot is at path index 10 (cell 2,3) and neighbor (2,2) at path index 8 is untested (value=0),
            # this method will return [(2, 2)] because it's untested and comes before in the path.
        """
        if path is None or path_index is None:
            print(f"[ERROR] get_unknow_neighbors_with_unexplored_side requires path and path_index")
            return None

        untested_neighbors = []

        print(f"\n[CHECK] Looking for UNTESTED neighbors (value=0) with lower path index around cell ({cell_row},{cell_col}) at path_index={path_index}")

        # Create a mapping from (row, col) to path index for quick lookup
        cell_to_path_index = {cell: idx for idx, cell in enumerate(path)}

        # Define neighbors in absolute grid directions
        # Format: (delta_row, delta_col, direction_name, side_bit_to_check)
        neighbors = [
            (-1, 0, 'North', 0b0010),  # North neighbor: check its South side (facing us)
            (1, 0, 'South', 0b1000),   # South neighbor: check its North side (facing us)
            (0, 1, 'East', 0b0001),    # East neighbor: check its West side (facing us)
            (0, -1, 'West', 0b0100)    # West neighbor: check its East side (facing us)
        ]

        for dr, dc, direction, side_bit in neighbors:
            neighbor_row = cell_row + dr
            neighbor_col = cell_col + dc

            # Check if neighbor is within bounds
            if not (0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols):
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Out of bounds")
                continue

            # Check if neighbor is UNTESTED (value == 0)
            if self.map[neighbor_row][neighbor_col] != 0:
                status = "visited" if self.map[neighbor_row][neighbor_col] == 1 else "blocked"
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Already tested (status={status}, value={self.map[neighbor_row][neighbor_col]})")
                continue

            # Check if neighbor is in the path
            neighbor_cell = (neighbor_row, neighbor_col)
            if neighbor_cell not in cell_to_path_index:
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Not in serpentine path")
                continue

            neighbor_path_index = cell_to_path_index[neighbor_cell]

            # Check if neighbor has LOWER path index (comes before in serpentine)
            if neighbor_path_index >= path_index:
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Path index {neighbor_path_index} >= current {path_index} (comes after)")
                continue

            # Get explored sides of the untested neighbor
            neighbor_sides = self.get_cell_sides_status(neighbor_row, neighbor_col)

            # Check if the side facing the current cell is NOT explored
            if not (neighbor_sides & side_bit):
                # This side is NOT explored yet!
                untested_neighbors.append((neighbor_row, neighbor_col))
                print(f"  {direction} ({neighbor_row},{neighbor_col}): ✓ FOUND! Untested (value=0), "
                      f"path_index={neighbor_path_index} < {path_index}, "
                      f"sides={bin(neighbor_sides)}, side {bin(side_bit)} facing current cell is UNEXPLORED")
            else:
                print(f"  {direction} ({neighbor_row},{neighbor_col}): Untested but side {bin(side_bit)} "
                      f"facing current cell already explored (sides={bin(neighbor_sides)})")

        if untested_neighbors:
            # print(f"[RESULT] Found {len(untested_neighbors)} untested neighbor(s) with lower path index: {untested_neighbors}\n")
            return untested_neighbors
        else:
            print(f"[RESULT] No untested neighbors with lower path index found for cell ({cell_row},{cell_col})\n")
            return None

    def get_cell_sides_status(self, row, col):
        """
        Get the exploration status of a cell's sides.

        Args:
            row: Row of the cell
            col: Column of the cell

        Returns:
            int: 4-bit value representing explored sides (0b0000 to 0b1111)
        """
        cell_key = (row, col)
        return self.explored_sides.get(cell_key, 0b0000)

    def print_sides_status(self, row, col):
        """
        Print human-readable status of explored sides for a cell.

        Args:
            row: Row of the cell
            col: Column of the cell
        """
        sides = self.get_cell_sides_status(row, col)

        # Map bits to side names
        sides_map = {
            0b1000: "Nord (↑)",
            0b0100: "Est (→)",
            0b0010: "Sud (↓)",
            0b0001: "Ovest (←)"
        }

        explored = [name for bit, name in sides_map.items() if sides & bit]

        print(f"[SIDES] Cell ({row},{col})")
        print(f"  Binary: {bin(sides)} | Decimal: {sides}")
        print(f"  Explored sides: {', '.join(explored) if explored else 'None'}")
        print(f"  Count: {bin(sides).count('1')}/4")

        if sides == 0b1111:
            print(f"  Status: FULLY EXPLORED (all sides attempted)")
        elif sides == 0b0000:
            print(f"  Status: UNEXPLORED (no attempts)")
        else:
            print(f"  Status: PARTIALLY EXPLORED")
    def set_origin(self, x, y, yaw=0.0, start_row=0, start_col=0):
        """
        Set the world coordinates (x, y, yaw) as the origin of the grid.
        The grid is aligned with the robot's initial orientation.
        Mark the starting cell as visited (value 1).

        Args:
            x: World X coordinate to set as origin
            y: World Y coordinate to set as origin
            yaw: Initial yaw orientation of robot in radians
            start_row: Row index of starting cell (default 0)
            start_col: Column index of starting cell (default 0)
        """
        self.origin_x = x
        self.origin_y = y
        self.origin_yaw = yaw
        self.start_cell = (start_row, start_col)

        # Mark starting cell as visited
        self.map[start_row][start_col] = 1

        print(f"[OK] Origin set at world coordinates ({x:.2f}, {y:.2f})")
        print(f"[OK] Initial orientation: {np.rad2deg(yaw):.1f}°")
        print(f"[OK] Starting cell ({start_row}, {start_col}) marked as visited")

    def update_position(self, x, y):
        """
        Update the map based on new world coordinates.
        Marks cell as visited only when robot center is inside the cell boundaries.
        Coordinates are rotated to align with the robot's initial orientation.

        Args:
            x: Current world X coordinate
            y: Current world Y coordinate

        Returns:
            tuple: (row, col) of the current cell, or None if out of bounds
        """
        # Calculate relative position from origin
        delta_x = x - self.origin_x
        delta_y = y - self.origin_y

        # Rotate coordinates to align with grid (inverse rotation)
        # Grid X-axis should align with robot's initial forward direction
        cos_yaw = np.cos(-self.origin_yaw)
        sin_yaw = np.sin(-self.origin_yaw)

        grid_x = delta_x * cos_yaw - delta_y * sin_yaw
        grid_y = delta_x * sin_yaw + delta_y * cos_yaw

        # Convert to grid coordinates by checking which cell contains the robot center
        # Grid: X -> columns (right), Y -> rows (forward)
        # Find the cell index by checking boundaries
        half_size = self.cell_size / 2.0

        # Calculate which cell the robot center is in
        col = None
        row = None

        for c in range(self.cols):
            cell_center_x = (c - self.start_cell[1]) * self.cell_size
            if abs(grid_x - cell_center_x) <= half_size:
                col = c
                break

        for r in range(self.rows):
            cell_center_y = (r - self.start_cell[0]) * self.cell_size
            if abs(grid_y - cell_center_y) <= half_size:
                row = r
                break

        # Check if we found a valid cell
        if row is not None and col is not None:
            # Check bounds (should always be true if above logic is correct)
            if 0 <= row < self.rows and 0 <= col < self.cols:
                # Mark cell as visited if not already
                if self.map[row][col] == 0:
                    self.map[row][col] = 1
                    print(f"[MAP] Cell ({row}, {col}) marked as visited (robot center inside)")

                self.current_cell = (row, col)
                return (row, col)

        # Robot center is not inside any cell (between cells or out of bounds)
        print(f"[INFO] Robot center at ({x:.2f}, {y:.2f}) -> grid ({grid_x:.2f}, {grid_y:.2f}) is between cells or out of bounds")
        return None

    def get_cell_from_world(self, x, y):
        """
        Convert world coordinates to grid cell without marking as visited.
        Coordinates are rotated to align with the robot's initial orientation.
        """
        delta_x = x - self.origin_x
        delta_y = y - self.origin_y

        # Rotate coordinates to align with grid (inverse rotation)
        cos_yaw = np.cos(-self.origin_yaw)
        sin_yaw = np.sin(-self.origin_yaw)

        grid_x = delta_x * cos_yaw - delta_y * sin_yaw
        grid_y = delta_x * sin_yaw + delta_y * cos_yaw

        col = self.start_cell[1] + int(grid_x / self.cell_size)
        row = self.start_cell[0] + int(grid_y / self.cell_size)

        return (row, col)

    def world_to_grid_cell(self, x, y):
        """
        Alias for get_cell_from_world for clearer semantics.
        Convert world coordinates to grid cell without marking as visited.
        """
        return self.get_cell_from_world(x, y)

    def get_cell_status(self, row, col):
        """
        Get the status of a cell in the map.

        Args:
            row: Row index
            col: Column index

        Returns:
            tuple: (cell_value, explored_sides) where cell_value is:
                   0 = unvisited
                   1 = visited and accessible
                   -1 = attempted but blocked
                   explored_sides is the 4-bit value representing explored sides.
                   Returns (None, None) if out of bounds.
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            cell_value = self.map[row][col]
            explored_sides = self.get_cell_sides_status(row, col)
            return cell_value, explored_sides
        return None, None

    def generate_serpentine_path(self):
        """
        Generate a serpentine (lawnmower) path through the grid.
        Pattern:
        1  2  3  4  5
        10 9  8  7  6
        11 12 13 14 15
        ...

        Returns:
            list: List of (row, col) tuples in order to visit
        """
        path = []
        for row in range(self.rows):
            if row % 2 == 0:
                # Even rows: left to right
                for col in range(self.cols):
                    path.append((row, col))
            else:
                # Odd rows: right to left
                for col in range(self.cols - 1, -1, -1):
                    path.append((row, col))
        return path

    def get_world_position_from_cell(self, row, col):
        """
        Convert grid cell to world coordinates (center of cell).
        Applies rotation to transform from grid frame to world frame.
        Cell (0,0) center is at the origin (robot's starting position).

        Args:
            row: Row index
            col: Column index

        Returns:
            tuple: (x, y) world coordinates, or None if out of bounds
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # Calculate offset from start cell in grid frame
            delta_row = row - self.start_cell[0]
            delta_col = col - self.start_cell[1]

            # Grid coordinates (center of cell)
            # Grid: X -> columns (right), Y -> rows (forward)
            # Note: No offset added because cell (0,0) center should be at origin
            grid_x = delta_col * self.cell_size
            grid_y = delta_row * self.cell_size

            # Rotate to world frame (forward rotation)
            cos_yaw = np.cos(self.origin_yaw)
            sin_yaw = np.sin(self.origin_yaw)

            world_delta_x = grid_x * cos_yaw - grid_y * sin_yaw
            world_delta_y = grid_x * sin_yaw + grid_y * cos_yaw

            # Add to origin
            x = self.origin_x + world_delta_x
            y = self.origin_y + world_delta_y

            return (x, y)
        return None

    def add_waypoint(self, x, y):
        """
        Register a waypoint at the given world coordinates.

        Args:
            x: World X coordinate
            y: World Y coordinate
        """
        self.waypoints.append((x, y))
        #print(f"[WAYPOINT] Registered waypoint #{len(self.waypoints)} at ({x:.3f}, {y:.3f})")

    def add_robot_position(self, x, y, movement_type='explore'):
        """
        Register a robot position to track its path.

        Args:
            x: World X coordinate
            y: World Y coordinate
            movement_type: 'explore' for exploring new cells, 'navigate' for graph navigation
        """
        self.robot_path.append((x, y, movement_type))
        #print(f"[PATH] Robot position #{len(self.robot_path)} recorded at ({x:.3f}, {y:.3f}) [{movement_type}]")

    def print_map(self):
        """Print the current state of the map."""
        print("\n--- Environment Map ---")
        for row in self.map:
            print(" ".join(str(cell) for cell in row))
        if hasattr(self, 'current_cell'):
            print(f"Current cell: {self.current_cell}")
        print("-----------------------\n")



