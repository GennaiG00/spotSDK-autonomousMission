import time

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.api.mission
import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.mission.client
import bosdyn.util
from bosdyn.api.graph_nav import recording_pb2
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.robot_command import RobotCommandBuilder

def _toggle_record(self):
    """toggle recording on/off. Initial state is OFF"""
    if self._recording_client is not None:
        recording_status = self._recording_client.get_record_status()

        if not recording_status.is_recording:
            # Start recording map
            start_recording_response = self._recording_client.start_recording_full()
            if start_recording_response.status != recording_pb2.StartRecordingResponse.STATUS_OK:
                print(f'Error starting recording (status = {start_recording_response.status}).')
                return False
            else:
                print('Recording started - Stop Recording mode active')
                print('Create and record actions for Autowalk')
                if not self.resumed_recording:
                    del self.walk.elements[:]
                    start_element = self._create_element(
                        'Start', start_recording_response.created_waypoint, isAction=False)
                    self.elements.append(start_element)

        else:
            # Stop recording map
            while True:
                try:
                    # For some reason it doesn't work the first time, no matter what
                    stop_status = self._recording_client.stop_recording()
                    if stop_status != recording_pb2.StopRecordingResponse.STATUS_NOT_READY_YET:
                        break
                except bosdyn.client.recording.NotReadyYetError:
                    time.sleep(0.1)

            if stop_status != recording_pb2.StopRecordingResponse.STATUS_OK:
                print(f'Error stopping recording (status = {stop_status}).')
                return False

            self.resumed_recording = True
            print('Recording stopped - Resume Recording mode available')
            print('Review and save Autowalk')


def _toggle_lease(self):
    """toggle lease acquisition. Initial state is acquired"""
    if self._lease_client is not None:
        if self._lease_keepalive is None:
            self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                                   return_at_exit=True)
            print('Lease acquired - Robot is now leased')
        else:
            self._lease_keepalive.shutdown()
            self._lease_keepalive = None
            print('Lease released - Robot is now unleased')


def _toggle_power(self):
    """toggle motor power. Initial state is OFF"""
    power_state = self._power_state()
    if power_state is None:
        print('Power state: Unknown')
        return
    if power_state == robot_state_proto.PowerState.STATE_OFF:
        self._try_grpc_async('powering-on', self._request_power_on)
        print('Robot powered on')
    else:
        self._try_grpc('powering-off', self._safe_power_off)
        print('Robot powered off')


# Robot command implementations
def _start_robot_command(self, desc, command_proto, end_time_secs=None):
    def _start_command():
        self._robot_command_client.robot_command(command=command_proto,
                                                 end_time_secs=end_time_secs)

    self._try_grpc(desc, _start_command)


def _self_right(self):
    self._start_robot_command('self_right', RobotCommandBuilder.selfright_command())


def _battery_change_pose(self):
    self._start_robot_command(
        'battery_change_pose',
        RobotCommandBuilder.battery_change_pose_command(
            dir_hint=basic_command_pb2.BatteryChangePoseCommand.Request.HINT_RIGHT))


def _sit(self):
    self._start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())


def _stand(self):
    self._start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())


def _move_forward(self):
    self._velocity_cmd_helper('move_forward', v_x=self.linear_velocity)


def _move_backward(self):
    self._velocity_cmd_helper('move_backward', v_x=-self.linear_velocity)


def _strafe_left(self):
    self._velocity_cmd_helper('strafe_left', v_y=self.linear_velocity)


def _strafe_right(self):
    self._velocity_cmd_helper('strafe_right', v_y=-self.linear_velocity)


def _turn_left(self):
    self._velocity_cmd_helper('turn_left', v_rot=self.angular_velocity)


def _turn_right(self):
    self._velocity_cmd_helper('turn_right', v_rot=-self.angular_velocity)


def _stop(self):
    self._start_robot_command('stop', RobotCommandBuilder.stop_command())


def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
    self._start_robot_command(
        desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
        end_time_secs=time.time() + self.command_duration)


def _pose_command(self):
    self._start_robot_command(
        '',
        RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            0, 0, 0, self._robot.get_frame_tree_snapshot(),
            params=RobotCommandBuilder.mobility_params(body_height=self.robot_height,
                                                       footprint_R_body=self.euler_angles),
            body_height=self.robot_height), end_time_secs=time.time() + 1)


def _request_power_on(self):
    request = PowerServiceProto.PowerCommandRequest.REQUEST_ON
    return self._power_client.power_command_async(request)


def _safe_power_off(self):
    self._start_robot_command('safe_power_off', RobotCommandBuilder.safe_power_off_command())


def _power_state(self):
    state = self.robot_state
    if not state:
        return None
    return state.power_state.motor_power_state