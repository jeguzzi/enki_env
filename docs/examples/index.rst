========
Examples
========

The following three simple example show how to create an environment, 
how to evaluate a baseline policy (written by hand), how to train and evaluate a RL policy, and how to generate a video.
In order, we will

- control a single robot
- control two robots of the same type with the same policy
- control two different robots

Each section in the examples is implemented in a module in the `example package <https://github.com/jeguzzi/enki_env/tree/main/enki_env/examples>`_ and can be run

   python -m enki_env.examples.<scenario>.<section>

like, e.g. ::
   
   python -m enki_env.examples.single.environment

   
In the documentation, we include one notebook per example that showcases the pipeline.

.. toctree::
   :maxdepth: 2

   single_robot
   same_robots
   different_robots
   
   
