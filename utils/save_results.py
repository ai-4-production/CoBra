import json
from objects import cells, order
from copy import copy, deepcopy


class SimulationResults:
    instances = []

    def __init__(self, sim_env):
        """Save results of a simulation run as dict schema. Can be used to save after all runs finished."""
        sim_results = copy(schema_simulation)
        sim_results["run_number"] = len(self.__class__.instances) + 1
        sim_results["seed_incoming_orders"] = sim_env.seed_incoming_orders
        sim_results["seed_machine_interruptions"] = sim_env.seed_machine_interruptions
        sim_results["simulation_results"] = sim_env.result

        # Fill cell schema
        for cell in cells.Cell.instances:
            cell_schema = deepcopy(schema_cells)
            cell_schema["id"] = cell.id
            cell_schema["level"] = cell.level
            if cell.parent:
                cell_schema["parent"] = cell.parent.id
            else:
                cell_schema["parent"] = None
            cell_schema["cell_results"] = cell.result

            # Fill agent schema
            for agent in cell.agents:
                agent_schema = deepcopy(schema_agents)
                agent_schema["ruleset"] = agent.ruleset.name.decode("UTF-8")
                agent_schema["agent_results"] = agent.result
                cell_schema["agents"].append(agent_schema)

            # Fill machine schema
            for machine in cell.machines:
                machine_schema = deepcopy(schema_machines)
                machine_schema["type"] = machine.performable_task.name.decode("UTF-8")
                machine_schema["machine_results"] = machine.result
                cell_schema["machines"].append(machine_schema)

            # Fill input buffer schema
            input_b_schema = deepcopy(schema_buffer)
            input_b_schema["type"] = "Input-Buffer"
            input_b_schema["capacity"] = cell.input_buffer.storage_capacity
            input_b_schema["buffer_results"] = cell.input_buffer.result
            cell_schema["buffer"].append(input_b_schema)

            # Fill output buffer schema
            output_b_schema = deepcopy(schema_buffer)
            output_b_schema["type"] = "Output-Buffer"
            output_b_schema["capacity"] = cell.output_buffer.storage_capacity
            output_b_schema["buffer_results"] = cell.output_buffer.result
            cell_schema["buffer"].append(output_b_schema)

            # Fill storage buffer schema
            storage_b_schema = deepcopy(schema_buffer)
            storage_b_schema["type"] = "Storage-Buffer"
            storage_b_schema["capacity"] = cell.storage.storage_capacity
            storage_b_schema["buffer_results"] = cell.storage.result
            cell_schema["buffer"].append(storage_b_schema)

            # Fill interface buffers schema
            for interface in cell.interfaces_in:
                interface_in_schema = deepcopy(schema_buffer)
                interface_in_schema["type"] = "Interface-Buffer Outgoing"
                interface_in_schema["capacity"] = interface.storage_capacity
                interface_in_schema["buffer_results"] = interface.result
                cell_schema["buffer"].append(interface_in_schema)

            for interface in cell.interfaces_out:
                interface_out_schema = deepcopy(schema_buffer)
                interface_out_schema["type"] = "Interface-Buffer Ingoing"
                interface_out_schema["capacity"] = interface.storage_capacity
                interface_out_schema["buffer_results"] = interface.result
                cell_schema["buffer"].append(interface_out_schema)

            sim_results["cells"].append(cell_schema)

        order_results = {}
        order_results["run_number"] = len(self.__class__.instances) + 1
        order_results["seed_incoming_orders"] = sim_env.seed_incoming_orders
        order_results["orders"] = []

        for item in order.Order.instances:
            item_schema = deepcopy(schema_items)
            item_schema["type"] = item.type.name.decode("UTF-8")
            item_schema["start"] = item.start
            item_schema["due_to"] = item.due_to
            item_schema["priority"] = int(item.priority)
            item_schema["item_results"] = item.result
            order_results["orders"].append(item_schema)

        self.results = sim_results
        self.order_results = order_results
        self.__class__.instances.append(self)


schema_simulation = json.loads("""{
                "run_number": null,
                "seed_incoming_orders": null,
                "seed_machine_interruptions": null,
                "simulation_results": null,
                "cells": []
            }""")

schema_cells = json.loads("""
        { 
          "id": null,
          "level": null,
          "parent": null,
          "cell_results": null,
          "agents": [],
          "machines": [],
          "buffer": []
        }
        """)

schema_agents = json.loads("""
            {
              "ruleset": null,
              "agent_results": null
            }
        """)

schema_machines = json.loads("""
            {
              "type": null,
              "machine_results": null
            }
        """)

schema_buffer = json.loads("""
            {
              "type": null,
              "capacity": null,
              "buffer_results": null
            }
        """)

schema_items = json.loads("""
            {
              "type": null,
              "start": null,
              "due_to": null,
              "priority":null,
              "item_results": null
            }
        """)