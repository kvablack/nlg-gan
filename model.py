from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """
    Abstract base class providing an interface to build and store a model with certain hyperparameters. Initializes and
    stores all placeholders, which are the starting points for the TensorFlow computational graph. Placeholder objects
    may then be retrieved to be replaced in a session and output objects to be evaluated in a session.

    Methods that must be overridden:
        :func _gen_placeholders: Populates the placeholder_tensors dictionary with all the model-specific placeholders
        :func _build_model: Builds the computational model and populates the output_tensors dictionary with any model-specific tensors that
            should be accessible from the training loop

    .. note::
        Subclasses should document the model-specific initialization hyperparameters, placeholders, and fetchable
        output tensors.
    """

    def __init__(self, **kwargs):
        """
        Initialize a Model's placeholders, outputs, and hyperparameters. Subclasses should document the model-specific
        parameters to be passed to `__init__`.
        :param kwargs: any necessary hyperparameters for the model
        """
        self.placeholders = {}
        self.outputs = {}
        self.hyperparameters = kwargs
        self._gen_placeholders()
        self._build_model()

    @abstractmethod
    def _gen_placeholders(self):
        """
        Populates the placeholder_tensors dictionary with all the model-specific placeholders
        """
        pass

    @abstractmethod
    def _build_model(self):
        """
        Builds the computational model and populates the output_tensors dictionary with any model-specific tensors that
        should be accessible from the training loop
        """
        pass

    def get_feed_dict(self, **kwargs):
        """
        Takes the names and desired values of placeholders and turns them into a feed_dict that can be passed into
        a TensorFlow session in order to substitute the placeholders with the desired values
        :param kwargs: (name, desired value) pairs for placeholders
        :return: a dictionary of (placeholder object, desired value) pairs that can be passed directly into a session
        as a feed_dict
        """
        feed_dict = {}
        for key, value in kwargs.items():
            feed_dict[self.placeholders[key]] = value
        return feed_dict

    def get_fetch_dict(self, *args):
        """
        Takes the names of some or all of this Model's output tensors and retrieves a dict of tensor objects that can be
        passed into a TensorFlow session (as the `fetches` argument) in order to evaluate those tensors in the
        computational graph and compute their values
        :param args: a list of the names of output tensors that are to be evaluated
        :return: a dictionary of (name, tensor object) pairs from the model that can be passed into a session as the
        fetches argument
        """
        return dict([(name, self.outputs[name]) for name in args])