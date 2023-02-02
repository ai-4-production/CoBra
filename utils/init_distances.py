from configs.config import configuration
import time 


def init_cell_dimensions(cells):
    """Calculate and set height, width for each cell of an simulation environment
    and add distances between all positions within the cells
    :param cells: (Dataframe) Cell setup as dataframe. Either created by setup or loaded from setup file.
    """

    # Get basic configurations
    base_height = configuration["DISTANCES"]["BASE_HEIGHT"]
    base_width = configuration["DISTANCES"]["BASE_WIDTH"]
    cells_distance = configuration["DISTANCES"]["DISTANCE_BETWEEN_CELLS"]
    safe_distance = configuration["DISTANCES"]["SAFE_DISTANCE"]

    cells["height"] = (2 + 2 * cells["Level"]) * base_height
    cells["width"] = cells["machine_obj"].str.len() * base_width

    def distance(start_pos, end_pos, cell, best_path):
        if (start_pos in cell.interfaces_in and end_pos in cell.interfaces_out) or (start_pos in cell.interfaces_out and end_pos in cell.interfaces_in):
            if start_pos.coordinates[0] == end_pos.coordinates[0]:
                return 6 * safe_distance + start_pos.lower_cell.width + start_pos.lower_cell.height
            else:
                return 4 * safe_distance + abs(start_pos.coordinates[0] - end_pos.coordinates[0]) + abs(start_pos.coordinates[1] - end_pos.coordinates[1])

        elif (start_pos == cell.input_buffer and end_pos == cell.output_buffer) or (start_pos == cell.output_buffer and end_pos == cell.input_buffer):
            return cell.height + 2 * abs(start_pos.coordinates[0] - best_path)

        else:
            return abs(start_pos.coordinates[0] - end_pos.coordinates[0]) + abs(start_pos.coordinates[1] - end_pos.coordinates[1])

    # Calculate distances in machine cells (Hierachy level 0)

    for index, column in cells.loc[cells["Level"] == 0].iterrows():
        cell = column["cell_obj"]
        cell.width = column["width"]
        cell.height = column["height"]
        column["input_obj"].coordinates = (column["width"] / 2, 0)
        column["output_obj"].coordinates = (column["width"] / 2, column["height"])
        column["storage_obj"].coordinates = (0, column["height"] / 2)

        for idx, machine in enumerate(column["machine_obj"]):
            machine.coordinates = ((idx+1) * base_width, column["height"] / 2)

        start = cell.possible_positions
        end = cell.possible_positions

        cell.distances = [(start_pos, end_pos, abs(start_pos.coordinates[0]-end_pos.coordinates[0]) + abs(start_pos.coordinates[1]-end_pos.coordinates[1])) for start_pos in start for end_pos in end]

    # Calculate distances in distribution cells

    for index, column in cells.loc[cells["Level"] != 0].iterrows():
        cell = column["cell_obj"]
        cell.width = base_width + (len(cell.childs) - 1) * cells_distance + sum([child.width for child in cell.childs])
        cell.height = column["height"]
        column["input_obj"].coordinates = (cell.width / 2, 0)
        column["output_obj"].coordinates = (cell.width / 2, cell.height)
        column["storage_obj"].coordinates = (0, cell.height / 2)

        x = base_width
        interface_x = cell.width/2
        best_path = 0
        best_path_distance = float("inf")

        for child in cell.childs:
            x_start = x
            x_end = x + child.width

            if abs((x_start - safe_distance) - interface_x) < best_path_distance:
                best_path = (x_start - safe_distance)
                best_path_distance = abs((x_start - safe_distance) - interface_x)

            if abs((x_end + safe_distance) - interface_x) < best_path_distance:
                best_path = (x_end + safe_distance)
                best_path_distance = abs((x_end + safe_distance) - interface_x)

            child.input_buffer.coordinates = ((x_end - x_start)/2, base_height)
            child.output_buffer.coordinates = ((x_end - x_start)/2, cell.height - base_height)
            x = x_end + cells_distance

        start = cell.possible_positions
        end = cell.possible_positions
    
        cell.distances = [(start_pos, end_pos, distance(start_pos, end_pos, cell, best_path)) for start_pos in start for end_pos in end]
        