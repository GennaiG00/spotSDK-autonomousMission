import os
import time
from bosdyn.api.graph_nav.graph_nav_pb2 import TravelParams
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api.graph_nav import graph_nav_pb2, recording_pb2
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

        print(f"[INFO] Navigating back to first waypoint (waypoint_0)...")
        nav_to_cmd_id = None
        is_finished = False

        travel_params = TravelParams()
        travel_params.lost_detector_strictness = 2

        while not is_finished:
            nav_to_cmd_id = self._graph_nav_client.navigate_to(first_waypoint.id, 1.0, command_id=nav_to_cmd_id)
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
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

    def navigate_to_waypoint(self, waypoint_id, timeout=1.0):
        """
        Navigate to a specific waypoint by ID.

        Args:
            waypoint_id: The unique ID of the waypoint to navigate to
            timeout: Timeout for navigation command

        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        self._graph_nav_client.download_graph()

        nav_to_cmd_id = None
        is_finished = False

        print(f"[NAV] Navigating to waypoint ID: {waypoint_id}")
        travel_params = TravelParams()
        travel_params.lost_detector_strictness = 2

        while not is_finished:
            nav_to_cmd_id = self._graph_nav_client.navigate_to(waypoint_id, 1.0, command_id=nav_to_cmd_id)
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            is_finished = self._check_success(nav_to_cmd_id)

        print(f'[OK] Arrived at waypoint {waypoint_id}')
        return True

