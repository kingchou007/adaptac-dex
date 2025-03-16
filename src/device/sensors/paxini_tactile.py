import serial
import warnings
import struct
import rospy

from holodex.utils.network import FloatArrayPublisher
from holodex.constants import *


class PaxiniTactileStream:
    """
    Args:
        serial_port
        baudrate
        vis
    """

    def __init__(self, serial_port_number, tactile_num, baudrate, vis=False):
        # Initializing ROS Node
        rospy.init_node("tactile_{}_stream".format(tactile_num))
        self.tactile_data_publisher = FloatArrayPublisher(
            publisher_name="/tactile_{}/raw_data".format(tactile_num)
        )
        self.id = tactile_num
        # Setting ROS frequency
        self.rate = rospy.Rate(TACTILE_FPS)

        # Disabling scientific notations
        np.set_printoptions(suppress=True)

        self.serial_port_number = serial_port_number
        self.baudrate = baudrate

        self.force_unit = 0.1
        self.vis = vis
        self.point_per_sensor = POINT_PER_SENSOR  # accoding to paxini
        self.force_dim_per_point = FORCE_DIM_PER_POINT  # accoding to paxini
        self.data_chunk_size = self.point_per_sensor * self.force_dim_per_point
        self.force_data_start_index = 3  # accoding to paxini
        self.full_data_chunk_size = 50  # accoding to paxini
        # indices in paxini is not increase order, we change according to map
        self.ip_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
        ]  # accoding to paxini
        self.dp_indices = [
            4,
            13,
            1,
            2,
            3,
            12,
            14,
            7,
            8,
            0,
            11,
            6,
            5,
            9,
            10,
        ]  # accoding to paxini

        self.read_type = "all"  # "all" or "each

        self.tip_name = PAXINI_FINGER_PART_NAMES["tip"]  # accoding to paxini
        self.pulp_name = PAXINI_FINGER_PART_NAMES["pulp"]  # accoding to paxini

        self.start_tag = []
        for group_id in PAXINI_GROUP_INFO.keys():  # read each group first
            for finger_part_id in PAXINI_FINGER_PART_INFO.keys():
                self.start_tag.append(
                    PAXINI_FINGER_PART_INFO[finger_part_id]
                    + PAXINI_GROUP_INFO[group_id]
                )

        # assert self.start_tag == [b"\xaa\xee", b"\xcc\xee", b"\xaa\xff", b"\xcc\xff"]

        self.raw_data_tag = [b'\xaa\xee', b'\xcc\xee', b'\xaa\xff', b'\xcc\xff']

        self.sensor_number = len(self.start_tag)

        print(
            f"Started the Paxini Tactile {tactile_num} stream on port: {serial_port_number} with baudrate: {baudrate}!"
        )

    def open(self):
        self.serial_port = serial.Serial(
            port=self.serial_port_number,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )

    def close(self):
        if self.serial_port:
            self.serial_port.close()

    def flushInput(self):
        if self.serial_port:
            self.serial_port.reset_input_buffer()

    def read_each_sensor(self, start_tag):
        self.serial_port.read_until(start_tag)
        received_data = self.serial_port.read(1)
        if received_data != b"\x01":
            warning_message = f"\ndata invalid,will return None, tag={start_tag} state = {received_data}"
            warnings.warn(warning_message)
            self.wrong_data_flag = True
        else:
            return (
                self.serial_port.read(size=self.data_chunk_size),
                self.wrong_data_flag,
            )

    def read_all_sensors(self, start_tag):
        # the order of raw data is following: [b'\xaa\xee', b'\xcc\xee', b'\xaa\xff', b'\xcc\xff']
        self.wrong_data_flag = False
        self.serial_port.read_until(start_tag)
        received_data = self.serial_port.read_until(start_tag)
        received_data = (
            received_data[-len(start_tag) :] + received_data[: -len(start_tag)]
        )
        retry_time = 1

        while len(received_data) != self.sensor_number * self.full_data_chunk_size:
            self.serial_port.read_until(start_tag)
            received_data = self.serial_port.read_until(start_tag)
            received_data = (
                received_data[-len(start_tag) :] + received_data[: -len(start_tag)]
            )
            # print(f"retry time: {retry_time}")
            retry_time += 1

        sequence_len = len(received_data)
        format_string = f"<{sequence_len}b"
        integer_values = struct.unpack(format_string, received_data)
        integer_lists = list(integer_values)
        integer_lists = np.array(integer_lists).reshape(
            self.sensor_number, self.full_data_chunk_size
        )
        flag_lists = [integer_list[2] for integer_list in integer_lists]
        if len(set(flag_lists)) != 1:
            self.wrong_data_flag = True

        return received_data

    def split_data(self, tactile_data):
        data = tactile_data[self.force_data_start_index:self.force_data_start_index+self.data_chunk_size]
        fx, fy, fz = data[0::3], data[1::3], data[2::3]
        return [fx, fy, fz]
    
    def process_data(self, tactile_data):
        if self.read_type == "all":
            sequence_len = len(tactile_data)
            if Z_TYPE == 'wrong':
                format_string = f"<{sequence_len}b"
                integer_values = struct.unpack(format_string, tactile_data)
                integer_lists = list(integer_values)
                integer_lists = np.array(integer_lists).reshape(
                    self.sensor_number, self.full_data_chunk_size
                )
                data_lists = [
                    np.array(
                        integer_list[
                            self.force_data_start_index : self.force_data_start_index
                            + self.data_chunk_size
                        ]
                    ).reshape(self.point_per_sensor, self.force_dim_per_point)
                    for integer_list in integer_lists
                ]
            elif Z_TYPE == 'right':
                data = {}
                # decode force of z direction, spilt fx, fy ,fz
                for i in range(len(self.raw_data_tag)):
                    tactile_data_each = tactile_data[i*self.full_data_chunk_size:(i+1)*self.full_data_chunk_size]
                    tactile_data_each = self.split_data(tactile_data_each)
                    tactile_data_each[0], tactile_data_each[1] = struct.unpack(f'<{len(tactile_data_each[0])}b', bytes(tactile_data_each[0])), struct.unpack(f'<{len(tactile_data_each[1])}b', bytes(tactile_data_each[1]))
                    tactile_data_each[2] = struct.unpack(f'<{len(tactile_data_each[2])}B', bytes(tactile_data_each[2]))
                    data[self.raw_data_tag[i]] = tactile_data_each
                
                reorder_data = []
                for tag in self.start_tag:
                    reorder_data.append(data[tag])

                data_lists = np.array(reorder_data)
                data_lists = np.transpose(reorder_data, (0, 2, 1))
            return data_lists
        elif self.read_type == "each":
            sequence_len = len(tactile_data[0])
            format_string = f"<{sequence_len}b"
            integer_values = struct.unpack(format_string, tactile_data[0])
            # assert (np.array(integer_values)<6).all()
            integer_list = list(integer_values)
            integer_list = np.array(integer_list).reshape(
                self.point_per_sensor, self.force_dim_per_point
            )
            return integer_list

    def transform_data_order(self, tactile_data):
        for i in range(len(tactile_data)):
            if self.tip_name in str(self.start_tag[i]):
                tactile_data[i] = tactile_data[i][self.ip_indices]
            elif self.pulp_name in str(self.start_tag[i]):
                tactile_data[i] = tactile_data[i][self.dp_indices]

        return tactile_data

    def get_data(self):
        self.wrong_data_flag = True
        read_step = 0
        processed_data_list = None

        # if self.id == 1:
        while self.wrong_data_flag and read_step < 1:
            self.wrong_data_flag = False
            self.open()
            self.flushInput()

            if self.read_type == "all":
                raw_data_list = self.read_all_sensors(self.raw_data_tag[0])
            elif self.read_type == "each":
                raw_data_list = list(map(self.read_each_sensor, self.raw_data_tag))

            if not self.wrong_data_flag:
                if self.read_type == "all":
                    processed_data_list = np.array(self.process_data(raw_data_list))
                elif self.read_type == "each":
                    processed_data_list = np.array(
                        list(map(self.process_data, raw_data_list))
                    )
                processed_data_list = self.transform_data_order(processed_data_list)
            else:
                print(self.serial_port_number,None)

            self.close()
            read_step += 1
        # else:
        #     processed_data_list = np.zeros((self.sensor_number, self.point_per_sensor, self.force_dim_per_point))

        return processed_data_list

    def stream(self):
        print("Starting stream!\n")
        # import time
        while True:
            # st = time.time()
            tactile_data = self.get_data()
            # print(self.serial_port_number, time.time()-st, tactile_data is None)
            if tactile_data is not None:
                # Publishing the tactile data
                self.tactile_data_publisher.publish(
                    tactile_data.flatten().tolist(), self.id
                )
            self.rate.sleep()


if __name__ == "__main__":
    tactile = PaxiniTactileStream(
        serial_port_number="/dev/ttyUSB0", tactile_num=1, baudrate=460800
    )
    # tactile.stream()
    import time

    while True:
        st = time.time()
        tactile_data = tactile.get_data()
        print(time.time() - st)
        # print(tactile_data)
        if tactile_data is None:
            print(tactile_data)
        # else:
        #     print(tactile_data.shape)
