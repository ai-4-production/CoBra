import sys


def _input(message, input_type=str, max=None):
    """Print message and wait for user input. Check if input is valid.
    :param message: (str) Message to print
    :param input_type: Valid input type. String as standard.
    :param max: The highest possible value if input type is numeric
    :return The first valid user input"""

    while True:
        try:
            input_value = input_type(input(message))
            if max:
                if max < input_value:
                    raise Exception
            return input_value
        except:
            print('That´s not a valid option!\n')
            pass


def yes_no_question(question: str):
    """Print an yes no question for the user and wait for an valid choice.
    :param question: (str) The question to print
    :return First valid boolean for the user input"""

    valid = {"1": True, "yes": True, "y": True, "ye": True, "ja": True, "0": False, "no": False, "n": False, "nein": False}
    while True:
        sys.stdout.write(question)
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")