# -*- coding: utf-8 -*-
import json
import configs.models


class RuleSet:
    instances = []

    def __init__(self, rules: dict, train_model=False):
        self.__class__.instances.append(self)
        self.id = rules['id']
        self.name = rules['name'].encode()
        self.description = rules['description'].encode()
        try:
            self.random = bool(rules['rules']['random'])
            self.seed = rules['rules']['seed']
        except:
            self.random = False
            self.seed = None
            pass

        try:
            self.numerical_criteria = rules['rules']['criteria']['numerical']
        except:
            self.numerical_criteria = []
            pass

        try:
            self.dynamic = rules['rules']['dynamic']
            self.reinforce_agent = eval("configs.models." + rules['rules']['reinforcement_agent'])
            if train_model:
                self.reinforce_agent.trainModel()
        except:
            self.dynamic = False
            self.reinforce_agent = None


def load_rulesets(train_model):
    """Load possible rulesets from json file and create an object for each"""

    rulesets_config = json.load(open("../configs/rulesets.json", encoding='utf-8'))

    # Create objects
    for rule in rulesets_config['rulesets']:
        RuleSet(rule, train_model)