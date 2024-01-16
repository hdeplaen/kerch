# coding=utf-8
from ._LearningSet import _LearningSet
def factory(dataset='gaussians', **kwargs) -> _LearningSet:
    r"""
    # TODO
    """
    def case_insensitive_getattr(obj, attr):
        for a in dir(obj):
            if a.lower() == attr.lower():
                return getattr(obj, a)
        return None

    dataset = dataset.replace("_","")
    dataset = dataset.replace(" ","")

    if dataset.lower() == "pima":
        dataset = "pimaindians"

    import kerch.data
    learning_set = case_insensitive_getattr(kerch.data, dataset)
    if learning_set is None:
        raise NameError("Invalid dataset.")

    return learning_set(*args, **kwargs)
