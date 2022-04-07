import json
import os

loaded_materials = None


def load_materials():
    """Load materials from setup file"""
    global loaded_materials

    path = os.path.realpath(__file__)
    directory = os.path.dirname(path).replace("objects", "configs")
    os.chdir(directory)
    loaded_materials = json.load(open("materials.json"))

