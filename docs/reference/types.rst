=====
Types
=====

.. currentmodule:: enki_env.types

.. autoprotocol:: Scenario

.. autoprotocol:: RewardFunction

.. autoprotocol:: InfoFunction


.. autoprotocol:: Termination

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

