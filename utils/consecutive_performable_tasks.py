def consecutive_performable_tasks(next_tasks, performable_tasks):
    """Takes an list of processing steps and calculate how many of these can be
    consecutively performed by the performable tasks attribute of a cell
    :param next_tasks: (list of Processing Step objects) List to check, work schedule
    :param performable_tasks: (list of tuple containing each processing step and the amount of machines that can perform these step)
    :return Number of consecutive tasks"""

    performable_tasks = [task for (task, amount) in performable_tasks if amount > 0]

    next_tasks = [task in performable_tasks for task in next_tasks]

    if False in next_tasks:
        return next_tasks.index(False)
    else:
        return len(next_tasks)
