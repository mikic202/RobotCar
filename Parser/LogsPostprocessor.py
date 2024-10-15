import os
import pandas as pd
import numpy as np

class LogPostprocessor:
    @staticmethod
    def run(logs_dir: str):
        for dir in LogPostprocessor.find_log_dirs(logs_dir):
            LogPostprocessor.parocess_csv_files_in_dir(dir)

    @staticmethod
    def find_log_dirs(logs_dir: str):
        return [ f.path for f in os.scandir(logs_dir) if f.is_dir() ]

    @staticmethod
    def parocess_csv_files_in_dir(dir: str):
        if not os.path.isfile(os.path.join(dir, "sensor.csv")) or not os.path.isfile(os.path.join(dir, "control.csv")):
            return

        output = pd.merge(LogPostprocessor.restructure_csv(os.path.join(dir, "control.csv")), LogPostprocessor.restructure_csv(os.path.join(dir, "sensor.csv")), on="iteration")
        output.to_csv(os.path.join(dir, "combined.csv"), index=False)

    @staticmethod
    def restructure_csv(input_file: str):
        input_df = pd.read_csv(input_file, header=None)
        data_point_names = np.unique(input_df.values[:, 0])
        column_names = ["iteration", *data_point_names]
        number_of_iterations = len(np.unique(input_df.values[:, -1]))
        columns = (
            np.array(
                *[
                    len(column_names)
                    * [
                        np.arange(0, number_of_iterations),
                    ]
                ]
            )
            .astype(float)
            .T
        )
        first_index = int(input_df.values[0, -1])
        for data in input_df.values:
            columns[int(data[-1]) - first_index, column_names.index(data[0])] = data[1]

        return pd.DataFrame(columns, columns=column_names)


if __name__ == "__main__":
    LogPostprocessor.run("/home/mikic202/inz_robot/RobotCar/log")