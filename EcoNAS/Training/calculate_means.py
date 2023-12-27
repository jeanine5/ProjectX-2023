"""

"""

def calculate_average_accuracy(archs: list):
    """
    Calculates the average accuracy of the given architectures
    :param archs: The list of architectures
    :return: Returns the average accuracy
    """
    avg_acc = 0
    for arch in archs:
        avg_acc += arch.objectives['accuracy']

    return avg_acc / len(archs)


def calculate_average_interpretability(archs: list):
    """
    Calculates the average interpretability of the given architectures
    :param archs: The list of architectures
    :return: Returns the average interpretability
    """
    avg_interpretability = 0
    for arch in archs:
        avg_interpretability += arch.objectives['introspectability']

    return avg_interpretability / len(archs)

def calculate_average_flops(archs: list):
    """
    Calculates the average FLOPs of the given architectures
    :param archs: The list of architectures
    :return: Returns the average FLOPs
    """
    avg_flops = 0
    for arch in archs:
        avg_flops += arch.objectives['flops']

    return avg_flops / len(archs)
