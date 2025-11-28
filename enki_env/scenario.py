from __future__ import annotations

from typing import Protocol

import pyenki


class Scenario(Protocol):
    """
    A scenario is a generator of world.

    It exposes a callable that accepts a random seed and
    returns a :py:class:`pyenki.World`.

    >>> world = scenario(seed=0)

    The callable has an optional argument to copy the
    random generator from another world, which is passed
    during the environment reset with ``seed=None``.

    Users have two ways to implement a scenario:

    1. The can write a callable that conforms to :py:meth:`Scenario.__call__`,
       where they *must* pass the random seed to the world and possibly
       copy the random generator, e.g.::

           def my_scenario(seed: int, copy_rng_from: pyenki.World = None) -> pyenki.World:
               world = pyenki.World(seed=seed)
               if copy_rng_from:
                   world.copy_random_generator(copy_rng_from)
               ...
               return world

    2. They can specialize a sub-class of :py:class:`BaseScenario`, implementing
       one or both of :py:meth:`BaseScenario.make` and :py:meth:`BaseScenario.init`.
       This class takes care of setting random seeds and generators in its implementation of
       :py:meth:`BaseScenario.__call__`.

       For example ::

           class MyScenario

               def make(self) -> pyenki.World:
                   return pyenki.World(radius=100)

               def init(self, pyenki.World: world) -> None:
                   robot = pyenki.Thymio2()
                   robot.angle = world.random_generator.uniform(0, math.pi * 2)
                   world.add_object(world)

    Users should use :py:attr:`pyenki.World.random_generator` for random sampling
    to ensure reproducibility.
    """

    def __call__(self,
                 seed: int,
                 /,
                 copy_rng_from: pyenki.World | None = None) -> pyenki.World:
        """
        Generates of a world with a given random seed.

        :seed:                  The random seed.
        :copy_rng_from:      Another world to inherit the random generator
            from, if provided.
        :returns:               The generated world.
        """


class BaseScenario:
    """
    Base class for implementations of :py:type:`enki_env.Scenario` protocol.

    Users can specialize sub-classes by implementing
    one or both of :py:meth:`BaseScenario.make` and :py:meth:`BaseScenario.init`.

    For example ::

        class MyScenario

            def make(self) -> pyenki.World:
                return pyenki.World(radius=100)

            def init(self, pyenki.World: world) -> None:
                robot = pyenki.Thymio2()
                robot.angle = world.random_generator.uniform(0, math.pi * 2)
                world.add_object(world)
    """

    def make(self) -> pyenki.World:
        """
        Virtual method that returns a world.

        The base implementation returns a empty world with no boundaries.
        Can be overridden by sub-classes to add a boundary.

        .. important::

           Should defer random sampling to :py:meth:`Scenario.init`.

        :seed:       the random seed.
        :returns:    the world.
        """
        return pyenki.World()

    def init(self, world: pyenki.World) -> None:
        """
        Virtual method that initializes a given world.

        The base implementation does nothing.
        Can be overridden by sub-classes to add objects.

        .. important::

           Any random sampling should use
           :py:attr:`pyenki.World.random_generator`
           to ensure reproducibility.

        :param      world:  The generated world
        """
        return

    def __call__(self,
                 seed: int,
                 /,
                 copy_rng_from: pyenki.World | None = None) -> pyenki.World:
        """
        Implements :py:type:`Scenario` protocol.

        1. calls :py:meth:`BaseScenario.make`
        2. sets the random seed and generator
        3. calls :py:meth:`BaseScenario.init`

        :seed:                  The random seed.
        :copy_rng_from:      Another world to inherit the random generator
            from, if provided.
        :returns:               The world.
        """
        world = self.make()
        world.random_seed = seed
        if copy_rng_from:
            world.copy_random_generator(copy_rng_from)
        self.init(world)
        return world
