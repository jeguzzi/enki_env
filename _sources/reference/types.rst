=====
Types
=====

.. currentmodule:: enki_env.types

.. autoprotocol:: Scenario

.. py:type:: Termination
   :canonical: collections.abc.Callable[[pyenki.DifferentialWheeled, pyenki.World], bool | None]

   A criterion to decide the success/failure of an episode for a given robot. Should return ``True`` for success, ``False`` for failure, and ``None`` if not yet decided.

   For example, for a task where a robot needs to travel along the positive x-direction, we may select failure when
   if exits some narrow bands and success when it travels further enough: 

   .. code:: Python

      def my_criterion(robot: pyenki.DifferentialWheeled,
                       world: pyenki.World) -> bool | None:
          if robot.position[1] > 100:
              return True
          if abs(robot.position[0]) > 10:
              return False
          return None

.. py:type:: Observation
   :canonical: dict[str, numpy.typing.NDArray[np.float64]]

   ``enki_env`` environments uses dictionaries of floating-point arrays as observations. 

.. py:type:: Action
   :canonical: numpy.typing.NDArray[np.float64]

   ``enki_env`` environments uses floating-point arrays as actions. 

.. py:type:: Info
   :canonical: dict[str, typing.Any]

   Generic info dictionary

.. py:type:: State
   :canonical: tuple[Array, ...]

   State arrays

.. py:type:: EpisodeStart
   :canonical: Array

   Array that flags episodes start.

.. py:type:: PathLike
   :canonical: os.PathLike[str] | str

   Anything that can be converted to a file path

.. autoprotocol::  Predictor

.. py:type:: PyTorchObs
   :canonical: torch.Tensor | dict[str, torch.Tensor]

   The type of observations in PyTorch

.. autoprotocol::  PyTorchPolicy

