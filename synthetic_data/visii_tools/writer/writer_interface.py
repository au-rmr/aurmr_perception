
class WriterInterface:
    """
    This is the interface that writers should adhere to
    """

    def __init__(self, verbosity):
        self._verbosity = verbosity

    def run(self, h5, cameras=None, pipeline=None, entity_list=None, **kwargs):
        """


        Args:
            h5: h5py.Group
            pipeline: WriterPipeline object
            entity_list: lisf of entities to export data for

        """
        raise NotImplementedError()
