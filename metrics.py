"""
Utility methods to compare the performance of the classifiers
"""

def true_positive(gold_standard, prediction):
    """
    number of outcomes where the model correctly predicts the positive class
    :return: # of true positives
    """
    tp = 0
    for gs, pred in zip(gold_standard, prediction):
        if gs == 1 and pred == 1:
            tp += 1
    return tp


def true_negative(gold_standard, prediction):
    """
    number of outcomes where the model correctly predicts the negative class
    :return: # of true negatives
    """
    tn = 0
    for gs, pred in zip(gold_standard, prediction):
        if gs == 0 and pred == 0:
            tn += 1
    return tn


def false_positive(gold_standard, prediction):
    """
    number of outcomes where the model incorrectly predicts the positive class
    :return: # of false positives
    """
    fp = 0
    for gs, pred in zip(gold_standard, prediction):
        if gs == 0 and pred == 1:
            fp += 1
    return fp


def false_negative(gold_standard, prediction):
    """
    number of outcomes where the model incorrectly predicts the negative class
    :param gold_standard:
    :param prediction:
    :return: # of false negatives
    """
    fn = 0
    for gs, pred in zip(gold_standard, prediction):
        if gs == 1 and pred == 0:
            fn += 1
    return fn


def accuracy(gold_standard, prediction):
    """
    describes the model's behaviour across all classes.
    If all of the classes are comparably significant, it is helpful.
    :param gold_standard:
    :param prediction:
    :return: accuracy score
    """
    tp = true_negative(gold_standard, prediction)
    tn = true_negative(gold_standard, prediction)
    fp = false_positive(gold_standard, prediction)
    fn = false_negative(gold_standard, prediction)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc


def precision(gold_standard, prediction):
    """
    the ability of the classifier not to label as positive a sample that is negative.
    :param gold_standard:
    :param prediction:
    :return: precision score
    """
    tp = true_negative(gold_standard, prediction)
    fp = false_positive(gold_standard, prediction)
    prc = tp / (tp + fp)
    return prc


def recall(gold_standard, prediction):
    """
    the ability of the classifier to find all the positive samples.
    :param gold_standard:
    :param prediction:
    :return: recall score
    """
    tp = true_positive(gold_standard, prediction)
    fn = false_negative(gold_standard, prediction)
    rcl = tp / (tp + fn)
    return rcl


def f1(gold_standard, prediction):
    """
    The F1 score can be interpreted as a harmonic mean of the precision and recall,
    The relative contribution of precision and recall to the F1 score are equal.
    :param gold_standard:
    :param prediction:
    :return:
    """
    p = precision(gold_standard, prediction)
    r = recall(gold_standard, prediction)
    f1s = 2 * p * r / (p + r)
    return f1s