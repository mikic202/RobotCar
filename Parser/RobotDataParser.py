from typing import Any


class RobotDataParser:
    @staticmethod
    def convert_sensor_data_to_dict(
        data: list[tuple[int, float]]
    ) -> list[dict[str, Any]]:
        return [{"angle": reading[1], "value": reading[0]} for reading in data]

    @staticmethod
    def convert_motor_data_to_dict(data: list[float]) -> list[dict[str, Any]]:
        return [
            {"control_name": motor, "value": value} for motor, value in enumerate(data)
        ]

    @staticmethod
    def convert_data_dict_to_csv_rovs(
        log_values: list[dict[str, Any]], sample_index: int
    ) -> list[list[Any]]:
        return [
            [*[value for value in reading.values()], sample_index]
            for reading in log_values
        ]
