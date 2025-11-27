============
Introduction
============

Enki_env lets you use `pyenki <https://jeguzzi.github.io/enki/>`_ with common ML-libraries by wrapping it in environments that are compatible with `Gymnasium <https://gymnasium.farama.org>`_, `PettingZoo <https://pettingzoo.farama.org>`_, and `PyTorchRl <https://docs.pytorch.org/rl>`_.

Users can setup environments for any task involving the ground robots implemented by ``pyenki``, i.e., e-pucks, marxbots and thymios, by combining two parts:

- a scenario that generates a world populated with robots and static objects,  
- a configuration for each group of robots that defines which observations to include, how to actuate actions, which rewards to assign, and so on.

In the simplest case, we control a single robot, for example an e-puck. For this, we generate a world that contains the robot and some object to interact with. We could define a task where the e-puck uses its 8 proximity sensors to turn towards a nearby object.

.. code-block:: python
   
   import math
   
   import enki_env
   import gymnasium
   import pyenki
   
   
   def scenario(seed):
       world = pyenki.World(seed)
       robot = pyenki.EPuck()
       robot.angle = world.random_generator.uniform(0, 2 * math.pi)
       world.add_object(robot)
       obj = pyenki.PhysicalObject(radius=10, height=10, mass=-1)
       obj.position = (10, 0)
       world.add_object(obj)
       return world
   
   
   def reward(robot, world):
       return -1.0 if math.cos(robot.angle) < 0.9 else 0.0
   
   
   env = gymnasium.make("Enki",
                        scenario=scenario,
                        config=enki_env.EPuckConfig(reward=reward)

The environment is now ready for training or for evaluation. For example, we can compute the reward collected by a random policy during an episode:

.. code-block:: python
   
   >>> env.unwrapped.rollout(max_steps=10).episode_reward
   -10.0


In the more general case, we control multiple robots, possibly of different types. Robots that share the same configuration are grouped together. For example, we could create an environment where two e-pucks use the camera while three other e-pucks use the proximity sensors.

.. code-block:: python

   import enki_env
   import pyenki
   
   
   def scenario(seed):
       world = pyenki.World(seed)
       rng = world.random_generator
       for _ in range(2):
           robot = pyenki.EPuck(camera=True)
           robot.position = (rng.uniform(-10, 10), rng.uniform(-10, 10))
           robot.name = 'e-puck-camera'
           world.add_object(robot)
       for _ in range(3):
           robot = pyenki.EPuck(camera=False)
           robot.position = (rng.uniform(-10, 10), rng.uniform(-10, 10))
           world.add_object(robot)
       return world
   
   
   config = enki_env.EPuckConfig()
   config_camera = enki_env.EPuckConfig()
   config_camera.observation.camera = True
   config_camera.observation.proximity_value = False
   configs = {'e-puck': config, 'e-puck-camera': config_camera}
   
   env = enki_env.parallel_env(scenario, configs)
   env.reset(seed=0)


In the environment, robots are identified by a string `<group>_<index>`.

.. code-block:: python

   >>> print(env.agents)
   ['e-puck_0', 'e-puck_1', 'e-puck_2', 'e-puck-camera_0', 'e-puck-camera_1']

Robots in the same group share the same action and observation spaces, reward function, and (when assigned) policy.

   >>> print(env.group_map)
   {'e-puck': ['e-puck_0', 'e-puck_1', 'e-puck_2'], 'e-puck-camera': ['e-puck-camera_0', 'e-puck-camera_1']}


Robots in different groups will instead apply different policies and receive rewards from possibly different reward functions.

.. seealso::

   More comprehensive :doc:`examples <examples/index>`.
