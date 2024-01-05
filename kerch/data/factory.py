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

    import kerch.data
    kernel = case_insensitive_getattr(kerch.data, dataset)
    if kernel is None:
        raise NameError("Invalid dataset.")

    return kernel(**kwargs)
