"""
The coordinates used for training, predicting, and visualization are handled here.
"""

from typing import Final

class CoordinateFrame:
    """
    Encapsulates the flow coordinates in flow direction (flow_coord), tangential to the flow direction
    (flow_normal_coord), and tangential to the flow plane (flow_plane_normal_coord).
    """
    def __init__(self, flow_coord: str, flow_normal_coord: str, flow_plane_normal_coord):

        self._allowed_coordinates: Final[list[str]] = ["X (m)", "Y (m)", "Z (m)"]

        self._flow_coord: str = "UNINITIALIZED"
        self._flow_normal_coord: str = "UNINITIALIZED"
        self._flow_plane_normal_coord: str = "UNINITIALIZED"

        self.__errors: list[Exception] = []

        try:
            self._allowed_coordinates.remove(flow_coord)
            self._flow_coord = flow_coord
        except ValueError as value_error:
            self.__errors.append(value_error)
            print(f"{value_error}\nThe first direction {flow_coord} is not available. Only {self._allowed_coordinates} can be chosen.")

        try:
            self._allowed_coordinates.remove(flow_normal_coord)
            self._flow_normal_coord = flow_normal_coord
        except ValueError as value_error:
            self.__errors.append(value_error)

        try:
            self._allowed_coordinates.remove(flow_plane_normal_coord)
            self._flow_plane_normal_coord = flow_plane_normal_coord
        except ValueError as value_error:
            self.__errors.append(value_error)

        if self.__errors:
            raise InitializationError("Some attributes are **NOT** Initialized!", self.__errors)

    def get_flow_plane(self) -> list[str]:
        """
        Retrieves the flow coordinates in primary and normal directions.
        Returns:
            List of flow_coord and flow_normal_coord.
        """
        return [self._flow_coord, self._flow_normal_coord]

    def get_flow_area(self) -> list[str]:
        """
        Retrieves the flow coordinates normal to the flow in and out of plane, respectively.
        Returns:
            List of flow_normal_coord and flow_plane_normal_coord.
        """
        return [self._flow_normal_coord, self._flow_plane_normal_coord]

    def get_flow_plane_normal(self) -> str:
        """
        Retrieves the normal coordinate of the flow plane.
        Returns:
            flow_plane_normal_coord
        """
        return self._flow_plane_normal_coord

    def __str__(self) -> str:
        return f"flow_coord: {self._flow_coord}\nflow_normal_coord: {self._flow_normal_coord}\nflow_plane_normal_coord: {self._flow_plane_normal_coord}"

class InitializationError(Exception):
    def __init__(self, message: str, errors: list[Exception]):
        super().__init__(f"{message}\nREASON: {[error for error in errors]}")


def module_test() -> None:
    coord_frame = CoordinateFrame("X (m)", "Y (m)", "Z (m)")

    print(coord_frame.get_flow_plane())

if __name__ == "__main__":
    module_test()