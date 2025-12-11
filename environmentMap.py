
class EnvironmentMap(object):
    def __init__(self, rows=12, cols=12, cell_size=2.0):
        self.map = [[0 for _ in range(12)] for _ in range(12)]
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size  # Size of each cell in meters
        self.origin_x = 0.0  # World X coordinate of grid origin
        self.origin_y = 0.0  # World Y coordinate of grid origin
        self.start_cell = (0, 0)  # Starting cell in grid coordinates

    def set_origin(self, x, y, start_row=0, start_col=0):
        """
        Set the world coordinates (x, y) as the origin of the grid.
        Mark the starting cell as visited (value 1).

        Args:
            x: World X coordinate to set as origin
            y: World Y coordinate to set as origin
            start_row: Row index of starting cell (default 0)
            start_col: Column index of starting cell (default 0)
        """
        self.origin_x = x
        self.origin_y = y
        self.start_cell = (start_row, start_col)

        # Mark starting cell as visited
        self.map[start_row][start_col] = 1

        print(f"[OK] Origin set at world coordinates ({x}, {y})")
        print(f"[OK] Starting cell ({start_row}, {start_col}) marked as visited")

    def update_position(self, x, y):
        """
        Update the map based on new world coordinates.
        Calculates which cell the robot is in and marks it as visited.

        Args:
            x: Current world X coordinate
            y: Current world Y coordinate

        Returns:
            tuple: (row, col) of the current cell, or None if out of bounds
        """
        # Calculate relative position from origin
        delta_x = x - self.origin_x
        delta_y = y - self.origin_y

        # Convert to grid coordinates
        # Adding start_cell offset to account for starting position
        col = self.start_cell[1] + int(delta_x / self.cell_size)
        row = self.start_cell[0] - int(delta_y / self.cell_size)  # Negative because Y increases upward but row increases downward

        # Check bounds
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # Mark cell as visited if not already
            if self.map[row][col] == 0:
                self.map[row][col] = 1
                print(f"[MAP] Cell ({row}, {col}) marked as visited")

            self.current_cell = (row, col)
            return (row, col)
        else:
            print(f"[WARNING] Position ({x}, {y}) is out of map bounds")
            return None

    def get_cell_from_world(self, x, y):
        """
        Convert world coordinates to grid cell without marking as visited.
        """
        delta_x = x - self.origin_x
        delta_y = y - self.origin_y

        col = self.start_cell[1] + int(delta_x / self.cell_size)
        row = self.start_cell[0] - int(delta_y / self.cell_size)

        return (row, col)

    def print_map(self):
        """Print the current state of the map."""
        print("\n--- Environment Map ---")
        for row in self.map:
            print(" ".join(str(cell) for cell in row))
        print(f"Current cell: {self.current_cell}")
        print("-----------------------\n")

