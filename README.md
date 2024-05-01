# ML Model Handoff Cookbook

## 1. Use Python packaging
The standard way of distributing python packages is using a ``pyproject.toml`` file. 
You can find information here: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ 
and an example here: https://github.com/pypa/sampleproject/blob/main/pyproject.toml.

When your project has a python.toml you are able to easily create a "wheel" of your project which allows it to be 
easily distributed and installed with all of its dependencies:

```bash
pip install build
python -m build -w .
```

This command will create a ``.whl`` file in the ``dist`` folder which can then be installed using pip. It has the name
format of ``[project.name]-[project.version].whl``.

### (a) Use pinned versions for your dependencies
This is a very important step for reproducible builds. When you use “pip install” by default it goes out 
and gets the latest version of the package that is compatible with all the other packages you have 
installed. If we only know what packages to install, it could potentially install newer versions of 
the packages that you use that are not compatible with your original code. In “pyproject.toml” you 
can designate which version of a dependency to use as follows:

```toml
[project]
dependencies = [
  "torch==2.3.0"
]
```

In this example we have pinned PyTorch to version 2.3.0, whenever your project is installed it will also install using 

### (b) Use Python 3.8
Epic’s ML framework includes Python 3.8. You should develop your model using an install of Python 3.8 and also specify 
Python 3.8 in the ``[project]`` section of the toml.

```toml
requires-python = "==3.8"
```

### (c) Use virtual environments and test with a fresh virtual environment
A virtual environment is a Python environment that is separate from the system environment. Before handing off your model,
you should test that the model works using a fresh virtual environment. The following is an example of creating a new 
virtual environment.

```
python3.8 -m venv venv
source venv/bin/activate
pip install path/to/my_project
# test that model loads here
```

## 2. Project organization

### (a) Python packages
The standard structure for a Python project is to use python packages. A Python package is a folder with a 
``__init__.py`` file, which can be blank. This will make it so the folder and all of the ``.py`` files are importable 
when the wheel is installed using pip. For an example see the ``mypackage`` folder, which is importable via the 
following after installing:

```python
from mypackage import model
```

### (b) Resource Files
Keep (large) resources like weights files separate from code. The code will be installed from the wheel, while the 
weights will be placed in a resources folder and loaded by their path.

### (c) Use Git

Git is the most popular version control system. You can use it to track changes that you make to your code. Here are 
some useful commands:

Create repository
```bash
git init 
```

Add file or changed file to tracked files
```bash
git add mypackage/model.py
```

Commit changes to tracked files
```bash
git commit -m "I made some changes"
```

I recommend that you always commit any changes before a training run and record the results of ``git describe`` for that
training run. That way you will always be able to return to a previously trained model, even if you have since made 
changes to the code.

## 3. Code Usability

### (a) Have a documented “entry point”
You can do this in a number of ways: You can have a serialized model that will work after ``torch.load()``. 
You can create a single function that accepts arguments for a weights file and optionally a hyperparameters 
configuration file (yml, json, toml, etc.). The important thing is that the model can be instantiated in a 
python line or two. Test this entry point above using the fresh virtual environment. It should be documented
in the source file. 

See the ``create_model`` example in ``myproject/model.py``.

### (b) Have a documented function for calling the model
Have a single function that performs inference. This function should ideally take a Pandas DataFrame with 
well-defined columns; meaning for each column there should be documentation of its:
 
 - Data type: int, float, bool, str, etc.
 - Optionality: whether None / NaN is an acceptable value
 - Where in Epic did this value come from (Database column)? Did it need special preprocessing or some
   kind of transformation or creation of a statistic? Ideally your function should accept values as
   they occur in the Epic CDR / database and perform the transformations inside the function.
	
Things that introduce difficulties:
 - Only providing a function that accepts (packed) tensors as inputs. In this case we need to figure out
   a) how to convert from the Epic data type of the value to your tensor input and b) the ordering of the values.
 - Not providing code or even replicable steps for pre-processing. Ideally if your data requires some kind of
   transformation you provide an input function that accepts the original data as it occurs in Epic, or you
   provide a function that performs the transform before the model is called. 

#### Image Models: Special considerations
Images come to the wrapper jpeg encoded, that is the only image format available. The input options for your model are:
 - a) byte[] containing the raw jpeg data
 - b) PIL.Image
 - d) tensor (using torchvision’s PILToTensor or similar framework)

If you need to do transforms / preprocessing it should either be in a single designated function that
accepts one of the above 3 formats and performs the transform, returning a value suitable for input to the
model… OR the designated input function for the model should accept one of the 3 above inputs and do the
transform itself before calling the model. If you are using PyTorch, I recommend that you use 
[torchvision](https://pytorch.org/vision/stable/transforms.html) for any preprocessing transforms.

See the ``transform`` function and ``predict`` method on the ``MyModel`` class in ``mypackage/model.py``.

### (c) PyTorch Model Serialization

When you save a model using PyTorch, follow the 
[recommended practice](https://pytorch.org/tutorials/beginner/saving_loading_models.html) and save your model's 
state_dict rather than the entire model. This makes it so that saved weights can be still be used if the model code has
been updated.

### (d) Parameterize your model

Use a configuration object like a dictionary that can be serialized to a file to set your model hyperparameters rather
than hard-coding those parameters in the model.

This is easiest to explain with an example:

Hard-coding (bad):
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(15, 100)
```

Parameterization (preferrable):

```python
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden1 = nn.Linear(config['in_features'], config['hidden1_features'])
```

An example of why this helps: An early version of your model is deployed, but later through more testing, you discover
that performance is better  when one of the layers has a different number of features. If that number is hard coded in
the model itself, as in the first example, in order to update you would need to re-build the wheel. In the second
example, only a configuration file and the new weights would need to be replaced.

There are also additional benefits in record-keeping during the training if you write the hyperparameters to the same 
directory as the checkpoints and weights.

See ``mypackage/model.py`` for an example of a parameterized model. 
