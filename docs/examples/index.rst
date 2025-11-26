========
Examples
========

The following examples show how to create an environment, 
how to evaluate a baseline policy (written by hand), how to train and evaluate a RL policy, and how to generate a video.
We repeat these four steps for three scenario, where we 

- control a single robot
- control two robots of the same type with the same policy
- control two different robots.

Each scenario is implemented as a little package ``enki_env.examples.<scenario>`` in `examples <https://github.com/jeguzzi/enki_env/tree/main/enki_env/examples>`_, subdividing the common steps in four modules, which can be run separately like ::

   python -m enki_env.examples.<scenario>.<part>

where ``<part>`` is one of ``environment``, ``baseline``, ``rl`` or ``video`` and ``<scenario>`` is one of ``single_robot``, ``same_robots``, or ``different_robots``.
   
In the following, we include one notebook per scenario.


.. toctree::
   :maxdepth: 2

   single_robot
   same_robots
   different_robots
   
   
