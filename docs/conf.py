# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'enki_env'
copyright = '2025, Jerome Guzzi'
author = 'Jerome Guzzi'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx', 'sphinx_toolbox.more_autodoc.autoprotocol']

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
    'Observation': 'Observation',
    'Termination': 'Termination',
    'Scenario': 'Scenario',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
