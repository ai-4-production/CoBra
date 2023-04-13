# -*- coding: utf-8 -*-
import json
import os


class ProcessingStep:
    instances = []
    dummy_processing_step = None  # Used for orders that have no processing step left

    def __init__(self, task_config: dict, hidden=False):
        self.id = task_config["id"]
        self.name = task_config["title"].encode()
        self.base_duration = task_config["base_duration"]

        self.hidden = hidden
        if hidden:
            self.__class__.dummy_processing_step = self

        self.__class__.instances.append(self)


def load_processing_steps():
    """Load possible processing steps from json file and create an object for each"""

    path = os.path.realpath(__file__)
    directory = os.path.dirname(path).replace("objects", "configs")
    os.chdir(directory)

    processing_steps = json.load(open("processing_steps_paper_2.json", encoding="UTF-8"))

    # Hidden dummy processing step for finished orders
    ProcessingStep({"id": -1, "title": "Order finished", "base_duration": 0}, hidden=True)

    for step in processing_steps['tasks']:
        ProcessingStep(step)