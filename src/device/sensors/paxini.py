import serial
import warnings
import struct
import rospy


from adaptac.dataset.gen_data.process_obs import get_processed_tactile_data


class PaxiniTactile:
    def __init__(
        self,
        tactile_num,
    ):
        self.num_tactiles = tactile_num

        if self.num_tactiles > 0:
            self.tactile_info, _, _, self.sensor_per_board = fetch_paxini_info()

        self.tactile_subscribers = []
        for tactile_num in range(self.num_tactiles):
            self.tactile_subscribers.append(
                TactileSubscriber(tactile_num=tactile_num + 1)
            )

    def get_tactile(
        self, state_data, tactile_rep_type="3d_canonical_data", tactile_frame="hand"
    ):
        tactiles = self._transform_tactiles()
        tactile_data = get_processed_tactile_data(
            np.array([tactiles]),
            state_data,
            tactile_rep_type=tactile_rep_type,
            tactile_frame=tactile_frame,
        )
        return tactile_data

    def _transform_tactiles(self):
        tactile_data = {}
        for tactile_num in range(self.num_tactiles):
            raw_datas = np.array(
                self.tactile_subscribers[tactile_num].get_data()
            ).reshape(self.sensor_per_board, POINT_PER_SENSOR, FORCE_DIM_PER_POINT)
            for tactile_id, raw_data in enumerate(raw_datas):
                tactile_data[self.tactile_info["id"][tactile_num + 1][tactile_id]] = (
                    raw_data
                )

        temp_data = []
        for sensor_name in tactile_data:
            temp_data.extend(tactile_data[sensor_name].copy().reshape(-1).tolist())
        self.raw_tactile_data_for_vis = temp_data.copy()
        print("max tactile data: ", np.max(np.abs(temp_data)))

        return tactile_data


if __name__ == "__main__":
    pass
