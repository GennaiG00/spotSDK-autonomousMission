import math
import os
import time
from bosdyn.api.graph_nav.graph_nav_pb2 import TravelParams
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api.graph_nav import graph_nav_pb2, recording_pb2, nav_pb2
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient, map_pb2
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.math_helpers import Quat, SE3Pose

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

    def _set_initial_localization_waypoint(self, robot_state_client):
        """Trigger localization to a waypoint."""

        last_waypoint = self.get_last_waypoint()
        last_waypoint_id = last_waypoint["id"]

        if not last_waypoint_id:
            # Failed to find the unique waypoint id.
            return False

        robot_state = robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = last_waypoint_id
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

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
            return resp
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

    def online_full_graph_download(self):
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Failed to download the graph.')
            return
        for edge in graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            try:
                self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                print(f'Failed to download edge snapshot: {edge.snapshot_id}')
                continue
        for waypoint in graph.waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                self._graph_nav_client.download_waypoint_snapshot(waypoint.snapshot_id)
            except Exception:
                print(f'Failed to download waypoint snapshot: {waypoint.snapshot_id}')
                continue

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
        elif status.status == 7:
            return True
        else:
            return False

    def navigate_to_first_waypoint(self, robot_state_client=None):
        """Navigate back to the first waypoint (waypoint_0)."""
        graph = self._graph_nav_client.download_graph()
        first_waypoint = None
        for waypoint in graph.waypoints:
            if waypoint.annotations.name == "waypoint_0":
                first_waypoint = waypoint
        if first_waypoint is None:
            print('No waypoint_0 found in the graph.')
            return False

        self._set_initial_localization_waypoint(robot_state_client)
        print(f"[INFO] Navigating back to first waypoint (waypoint_0)...")
        nav_to_cmd_id = None
        is_finished = False

        travel_params = TravelParams()
        travel_params.lost_detector_strictness = 2

        while not is_finished:
            nav_to_cmd_id = self._graph_nav_client.navigate_to(first_waypoint.id, 1.0, command_id=nav_to_cmd_id)
            time.sleep(1)
            is_finished = self._check_success(nav_to_cmd_id)

        print("[OK] Arrived at first waypoint")
        return True

    def get_waypoint_list(self):
        """
        Get the list of all waypoints in the current graph.

        Returns:
            list: List of waypoint objects from the graph
        """
        graph = self._graph_nav_client.download_graph()
        if not graph or len(graph.waypoints) == 0:
            return []
        return list(graph.waypoints)

    def get_waypoint_details_list(self):
        """
        Get detailed information about all waypoints in the current graph.

        Returns:
            list: List of dictionaries with waypoint details:
                - 'id': Waypoint unique ID
                - 'name': Waypoint name (e.g., 'waypoint_0')
                - 'x': X position in world coordinates
                - 'y': Y position in world coordinates
                - 'z': Z position in world coordinates
                - 'waypoint_obj': Full waypoint object

        Example:
            waypoints = recording.get_waypoint_details_list()
            for wp in waypoints:
                print(f"Waypoint {wp['name']} at ({wp['x']:.2f}, {wp['y']:.2f})")
        """
        graph = self._graph_nav_client.download_graph()
        if not graph or len(graph.waypoints) == 0:
            print("[WAYPOINTS] No waypoints found in graph")
            return []

        waypoint_details = []

        for waypoint in graph.waypoints:
            # Extract position from waypoint_tform_ko (waypoint transform from kinematic odometry)
            transform = waypoint.waypoint_tform_ko

            details = {
                'id': waypoint.id,
                'name': waypoint.annotations.name if waypoint.annotations.name else 'unnamed',
                'x': transform.position.x,
                'y': transform.position.y,
                'z': transform.position.z,
                'waypoint_obj': waypoint
            }
            waypoint_details.append(details)

        print(f"[WAYPOINTS] Found {len(waypoint_details)} waypoints in graph:")
        for wp in waypoint_details:
            print(f"  - {wp['name']}: ID={wp['id']}, pos=({wp['x']:.3f}, {wp['y']:.3f}, {wp['z']:.3f})")

        return waypoint_details

    def get_last_waypoint(self):
        """
        Trova l'ultimo waypoint creato (quello con il numero più alto nel nome).

        Analizza tutti i waypoint nel grafo e restituisce quello con il numero più alto
        nel nome (es. waypoint_22 è l'ultimo se ci sono waypoint_0, waypoint_1, ..., waypoint_22).

        Returns:
            dict or None: Dizionario con dettagli dell'ultimo waypoint:
                - 'id': Waypoint unique ID
                - 'name': Waypoint name (e.g., 'waypoint_22')
                - 'number': Numero del waypoint (es. 22)
                - 'x': X position in world coordinates
                - 'y': Y position in world coordinates
                - 'z': Z position in world coordinates
                - 'waypoint_obj': Full waypoint object
                Oppure None se non ci sono waypoint nel grafo.

        Example:
            last_wp = recording.get_last_waypoint()
            if last_wp:
                print(f"Ultimo waypoint: {last_wp['name']} (#{last_wp['number']})")
                print(f"Posizione: ({last_wp['x']:.2f}, {last_wp['y']:.2f})")
        """
        graph = self._graph_nav_client.download_graph()
        if not graph or len(graph.waypoints) == 0:
            print("[LAST_WP] No waypoints found in graph")
            return None

        last_waypoint = None
        max_number = -1

        for waypoint in graph.waypoints:
            # Estrai il numero dal nome del waypoint (es. "waypoint_22" -> 22)
            name = waypoint.annotations.name if waypoint.annotations.name else 'unnamed'

            try:
                # Prova a estrarre il numero dopo "waypoint_"
                if name.startswith('waypoint_'):
                    number = int(name.split('_')[-1])

                    if number > max_number:
                        max_number = number
                        transform = waypoint.waypoint_tform_ko

                        last_waypoint = {
                            'id': waypoint.id,
                            'name': name,
                            'number': number,
                            'x': transform.position.x,
                            'y': transform.position.y,
                            'z': transform.position.z,
                            'waypoint_obj': waypoint
                        }
            except (ValueError, IndexError):
                # Salta waypoint con nomi non standard
                continue

        if last_waypoint:
            print(f"[LAST_WP] ✓ Ultimo waypoint: {last_waypoint['name']} (#{last_waypoint['number']})")
            print(f"[LAST_WP] Posizione: ({last_waypoint['x']:.3f}, {last_waypoint['y']:.3f}, {last_waypoint['z']:.3f})")
            print(f"[LAST_WP] ID: {last_waypoint['id']}")
        else:
            print(f"[LAST_WP] ⚠️ Nessun waypoint con formato standard 'waypoint_N' trovato")

        return last_waypoint

    def find_nearest_waypoint_to_position(self, target_x, target_y):
        """
        Trova il waypoint più vicino a una posizione world (x, y) specificata.

        Args:
            target_x: Coordinata X world della posizione target
            target_y: Coordinata Y world della posizione target

        Returns:
            dict or None: Dizionario con dettagli del waypoint più vicino:
                - 'id': ID del waypoint
                - 'name': Nome del waypoint
                - 'x', 'y', 'z': Posizione
                - 'distance': Distanza euclidea dalla posizione target
                - 'waypoint_obj': Oggetto waypoint completo
                Oppure None se non ci sono waypoint

        Example:
            # Trova waypoint più vicino al centro di una cella
            cell_center_x, cell_center_y = env.get_world_position_from_cell(row, col)
            nearest = recording.find_nearest_waypoint_to_position(cell_center_x, cell_center_y)
            if nearest:
                print(f"Waypoint più vicino: {nearest['name']} a {nearest['distance']:.2f}m")
        """
        import math

        waypoints = self.get_waypoint_details_list()
        if not waypoints:
            print("[NEAREST_WP] No waypoints available in graph")
            return None

        print(f"\n[NEAREST_WP] Ricerca waypoint più vicino a ({target_x:.3f}, {target_y:.3f})")

        min_distance = float('inf')
        nearest_waypoint = None

        for wp in waypoints:
            # Calcola distanza euclidea (ignorando Z per semplicità)
            distance = math.sqrt((wp['x'] - target_x)**2 + (wp['y'] - target_y)**2)

            print(f"  {wp['name']}: ({wp['x']:.3f}, {wp['y']:.3f}) - distanza: {distance:.3f}m")

            if distance < min_distance:
                min_distance = distance
                nearest_waypoint = wp.copy()
                nearest_waypoint['distance'] = distance

        if nearest_waypoint:
            print(f"[NEAREST_WP] ✓ Waypoint più vicino: {nearest_waypoint['name']} "
                  f"a {nearest_waypoint['distance']:.3f}m")

        return nearest_waypoint

    def navigate_to_waypoint(self, waypoint_id, robot_state_client=None):
        """
        Navigate to a specific waypoint by ID.

        Args:
            waypoint_id: The unique ID of the waypoint to navigate to
            timeout: Timeout for navigation command

        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        self._graph_nav_client.download_graph()
        self._set_initial_localization_waypoint(robot_state_client)

        nav_to_cmd_id = None
        is_finished = False

        print(f"[NAV] Navigating to waypoint ID: {waypoint_id}")
        travel_params = TravelParams()
        travel_params.lost_detector_strictness = 2

        while not is_finished:
            nav_to_cmd_id = self._graph_nav_client.navigate_to(waypoint_id, 1.0, command_id=nav_to_cmd_id)
            time.sleep(1)  # Sleep for half a second to allow for command execution.
            is_finished = self._check_success(nav_to_cmd_id)

        return is_finished

