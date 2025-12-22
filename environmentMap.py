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

    def get_cell_status(self, row, col):
        """
        Get the status of a cell in the map.

        Args:
            row: Row index
            col: Column index

        Returns:
            int: Cell value (0=unvisited, 1=visited), or None if out of bounds
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.map[row][col]
        return None

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

    def print_map(self):
        """Print the current state of the map."""
        print("\n--- Environment Map ---")
        for row in self.map:
            print(" ".join(str(cell) for cell in row))
        if hasattr(self, 'current_cell'):
            print(f"Current cell: {self.current_cell}")
        print("-----------------------\n")

