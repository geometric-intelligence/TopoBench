from modules.transforms.liftings.lifting import GraphLifting


class Graph2CombinatorialLifting(GraphLifting):
    r"""Abstract class for lifting graphs to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2combinatorial"
