# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from sphinx.addnodes import pending_xref

project = 'enki_env'
copyright = '2025, Jerome Guzzi'
author = 'Jerome Guzzi'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['nbsphinx', 'sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx_toolbox.more_autodoc.autoprotocol']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pyenki': ('https://jeguzzi.github.io/enki/', None),
    'gymnasium': ('https://gymnasium.farama.org', None),
    'pettingzoo': ('https://pettingzoo.farama.org', None),
    'torch': ('https://docs.pytorch.org/docs/stable', None),
    'stable_baselines3':
    ('https://stable-baselines3.readthedocs.io/en/master/', None),
    'onnxruntime': ('https://onnxruntime.ai/docs/api/python', None),
    'benchmarl': ('https://benchmarl.readthedocs.io/en/latest', None),
    'torchrl': ('https://docs.pytorch.org/rl/stable', None)
}

autodoc_default_options = {
    'exclude-members': '__weakref__, __new__'
}

autodoc_type_aliases = {
    'Info': 'Info',
    'Action': 'Action',
    'Array': 'Array',
    'Observation': 'Observation',
    'Termination': 'Termination',
    'Scenario': 'Scenario',
    'PyTorchObs': 'PyTorchObs',
    'Predictor': 'Predictor',
    'BoolArray': 'BoolArray'
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

_replace = {
    "np.": "numpy.",
    "collections.abc.": "",
    "typing.": "",
    "gym.": "gymnasium.",
}

_types = [
    'Array',
    'Action',
    'Observation',
    'pyenki.Controller',
    'Info',
    'PyTorchObs',
    'BoolArray'
]

_attrs = [
    'numpy.uint8', 'numpy.float64', 'numpy.float32', 'numpy.int64',
    'numpy.int32', 'numpy.bool_', "np.float64"
]

_data = ['numpy.typing.NDArray', 'numpy.typing.ArrayLike']

_meth = []

aliases = {
    "Sequence": "collections.abc.Sequence",
    "Callable": "collections.abc.Callable",
    "SupportsFloat": "typing.SupportsFloat",
    "SupportsInt": "typing.SupportsInt",
    "Unpack": "typing.Unpack",
    "Annotated": "typing.Annotated",
    "Any": "typing.Any",
    "PettingZooWrapper": "torchrl.envs.libs.pettingzoo.PettingZooWrapper",
    "gym.spaces.Box": "gymnasium.spaces.Box",
    "gym.spaces.Dict": "gymnasium.spaces.Dict",
    "gym.Space": "gymnasium.spaces.Space",
    "np.float64": "numpy.float64",
}

def f_docstring(app, what, name, obj, options, lines):
    for i, _ in enumerate(lines):
        if 'self' in lines[i]:
            import re

            lines[i] = re.sub(r"self: (\w+\.?)+,?\s*", "", lines[i])
        for k, v in _replace.items():
            if k in lines[i]:
                lines[i] = lines[i].replace(k, v)


def f_signature(app, what, name, obj, options, signature, return_annotation):
    if signature:
        import re

        signature = re.sub(r"self: (\w+\.?)+,?\s*", "", signature)
        for k, v in _replace.items():
            if k in signature:
                signature = signature.replace(k, v)
    if return_annotation:
        for k, v in _replace.items():
            if k in return_annotation:
                return_annotation = return_annotation.replace(k, v)
    return (signature, return_annotation)


def resolve_internal_aliases(app, doctree):
    pending_xrefs = doctree.traverse(condition=pending_xref)
    for node in pending_xrefs:
        if node['refdomain'] == "py":
            if node['reftarget'] in _types:
                node["reftype"] = "type"
            elif node['reftarget'] in _attrs:
                node["reftype"] = "attr"
            elif node['reftarget'] in _data:
                node["reftype"] = "data"
            elif node['reftarget'] in _meth:
                node["reftype"] = "meth"
        alias = node.get('reftarget', None)
        if alias is not None and alias in aliases:
            node['reftarget'] = aliases[alias]


def setup(app):
    app.connect('doctree-read', resolve_internal_aliases)
    app.connect('autodoc-process-docstring', f_docstring)
    app.connect('autodoc-process-signature', f_signature)
