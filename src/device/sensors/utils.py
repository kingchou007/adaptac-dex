from holodex.constants import *


def fetch_paxini_info():
    sensor_info = {}

    start_tags = []
    for group_id in PAXINI_GROUP_INFO:  # read each group first
        for finger_part_id in PAXINI_FINGER_PART_INFO:
            start_tags.append(
                PAXINI_FINGER_PART_INFO[finger_part_id] + PAXINI_GROUP_INFO[group_id]
            )

    # get finger name which has tactile
    sensor_info = {}
    sensor_info["id"] = {}
    sensor_info["finger_num"] = len(PAXINI_LEAPHAND)
    sensor_info["sensor_per_finger"] = len(PAXINI_FINGER_PART_INFO)
    sensor_info["sensor_type"] = []

    serial_port_numbers = []
    for finger in PAXINI_LEAPHAND.keys():
        cur_serial_port_number = PAXINI_LEAPHAND[finger]["serial_port_number"]
        serial_port_numbers.append(cur_serial_port_number)
    serial_port_numbers = list(set(serial_port_numbers))
    serial_port_numbers.sort()

    # for each port, port has to be consistent with defined, and same with collection and deploy
    for serial_port_number in serial_port_numbers:
        cur_id = int(serial_port_number[serial_port_number.find("USB") + 3]) + 1
        if cur_id not in sensor_info["id"]:
            sensor_info["id"][cur_id] = []

        for start_tag in start_tags:
            for finger in PAXINI_LEAPHAND:
                if PAXINI_LEAPHAND[finger]["serial_port_number"] == serial_port_number:
                    current_group_id = PAXINI_LEAPHAND[finger]["group_id"]
                    for finger_part_id in PAXINI_FINGER_PART_INFO:
                        if (
                            PAXINI_FINGER_PART_INFO[finger_part_id]
                            + PAXINI_GROUP_INFO[current_group_id]
                            == start_tag
                        ):
                            sensor_info["id"][cur_id].append(
                                finger + "_" + finger_part_id
                            )
                            if "tip" in finger_part_id:
                                sensor_info["sensor_type"].append("IP")
                            elif "pulp" in finger_part_id:
                                sensor_info["sensor_type"].append("DP")

    tactile_topic = []
    raw_data = {}
    for cur_serial_port_number in serial_port_numbers:
        idx = int(cur_serial_port_number[cur_serial_port_number.find("USB") + 3])
        tactile_topic.append("/tactile_{}/raw_data".format(idx + 1))
        raw_data[idx + 1] = None

    sensor_per_board = len(PAXINI_FINGER_PART_INFO) * len(PAXINI_GROUP_INFO)

    return sensor_info, tactile_topic, raw_data, sensor_per_board