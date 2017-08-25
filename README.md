## tensorflow_projects

For machine learning models in TensorFlow.

`plot.py` is a utility module providing a class for asynchronous plotting with matplotlib. See `basic_lstm.py` for example usage. Feel free to modify it to add more advanced functionality, but **update the docstrings** if you do.

`model.py` contains an abstract base class, `Model`, which provides a generic interface that should work with any machine learning architecture. See `basic_lstm/model.py` for an example subclass implementation.

Additional projects should be placed in subdirectories that include, at minimum:
1. A file, possibly empty, called `__init__.py` (in order for Python to recognize the directory as a package)
2. A concrete subclass implementation of `Model` that provides the methods to build the project-specific computational graph. Again, see `basic_lstm/model.py` for an example (although docstrings that robust are optional).

However, **do not** put any training scripts that actually evaluate the computational graph in the project subdirectories. Instead, put them in the top-level directory in a module with the same name as the package (see `basic_lstm.py`). 
