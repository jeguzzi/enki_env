============
Base classes
============

.. currentmodule:: enki_env

Actions
=======

Abstract base class
-------------------

.. autoclass:: enki_env.config.ActionConfig
   :members:

Robot base class
----------------

.. autoclass:: enki_env.robots.DifferentialDriveAction
   :members:
   :show-inheritance:
   :exclude-members: actuate, space

Observations
============

Abstract base class
-------------------

.. autoclass:: enki_env.config.ObservationConfig
   :members:

Robot base class
----------------

.. autoclass:: enki_env.robots.DifferentialDriveObservation
   :members:
   :exclude-members: actuate, space
   :show-inheritance:

Reward
======

Abstract base class
-------------------

.. autoclass:: enki_env.config.RewardConfig
   :members:

Concrete classes
----------------

.. autoclass:: enki_env.config.ConstRewardConfig
   :show-inheritance:

Info
====

Abstract base class
-------------------

.. autoclass:: enki_env.config.InfoConfig
   :members:

Concrete classes
----------------

.. autoclass:: enki_env.config.EmptyInfoConfig
   :show-inheritance:
