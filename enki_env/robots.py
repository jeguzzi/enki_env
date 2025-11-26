from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import numpy.typing

from .config import ActionConfig, GroupConfig, ObservationConfig
from .types import Action, Observation

if TYPE_CHECKING:
    from pyenki import DifferentialWheeled, EPuck, Marxbot, Thymio2


@dc.dataclass
class DifferentialDriveAction(ActionConfig):
    """
    Action configuration common to all robots

    Actions are given by two numbers in [-1, 1]
    which are re-scaled and actuated either as
    wheel speeds or accelerations.

    If either ``fix_orientation=True`` or ``fix_position=True``
    actions are a single number in [-1, 1] that get rescaled and
    then actuated as the sum/difference of wheel speeds or accelerations.
    """

    max_speed: float = 0
    """maximal wheel speed in cm/s"""
    max_acceleration: float = 1
    """
    maximal wheel acceleration in cm/s^2; only relevant if ``acceleration=True``
    """
    fix_orientation: bool = False
    """
    whether to nullify the angular components. If set, it reduces the dimension
    of the action space by one.
    """
    fix_position: bool = False
    """
    whether to nullify the linear components. If set, it reduces the dimension
    of the action space by one.
    """
    acceleration: bool = False
    """whether actions are wheel accelerations or speeds"""
    dtype: type[np.floating[Any]] = np.float64
    """the dtype of actions"""

    def actuate(self, act: Action, robot: DifferentialWheeled,
                dt: float) -> None:
        if self.fix_orientation:
            left, right = (act[0], act[0])
        elif self.fix_position:
            left, right = (-act[0], act[0])
        else:
            left, right = act
        if self.acceleration:
            robot.left_wheel_target_speed = np.clip(
                robot.left_wheel_encoder_speed +
                left * self.max_acceleration * dt, -self.max_speed,
                self.max_speed)
            robot.right_wheel_target_speed = np.clip(
                robot.right_wheel_encoder_speed +
                right * self.max_acceleration * dt, -self.max_speed,
                self.max_speed)
        else:
            robot.left_wheel_target_speed = left * self.max_speed
            robot.right_wheel_target_speed = right * self.max_speed

    @property
    def space(self) -> gym.spaces.Box:
        dim = 2 - sum((self.fix_orientation, self.fix_position))
        return gym.spaces.Box(low=-1, high=1, shape=(dim, ), dtype=self.dtype)


@dc.dataclass
class DifferentialDriveObservation(ObservationConfig):
    """
    Observation configuration common to all robots
    """
    max_speed: float = 0
    """maximal wheel speed in cm/s"""
    speed: bool = False
    """
    If set, observations will contain field ``"wheel_speeds"``
    associated with the current speed of the wheels (an array of length 2).
    """
    normalize: bool = True
    """whether to normalize the observation range."""
    dtype: type[np.floating[Any]] = np.float64
    """the type of floating point fields"""

    def get(self, robot: DifferentialWheeled) -> Observation:
        ks = {}
        if self.speed:
            ks['wheel_speeds'] = np.array([
                robot.left_wheel_encoder_speed, robot.right_wheel_encoder_speed
            ],
                                          dtype=self.dtype)
            if self.normalize:
                ks['wheel_speeds'] /= self.max_speed
        return ks

    @property
    def space(self) -> gym.spaces.Dict:
        ks: dict[str, gym.Space[Any]] = {}
        if self.speed:
            ks['wheel_speeds'] = gym.spaces.Box(
                low=-1 if self.normalize else -self.max_speed,
                high=1 if self.normalize else self.max_speed,
                shape=(2, ),
                dtype=self.dtype)
        return gym.spaces.Dict(ks)


@dc.dataclass
class MarxbotAction(DifferentialDriveAction):
    """
    Action configuration for :py:class:`pyenki.Marxbot`.

    Uses by default ``max_speed=30``.
    """
    max_speed: float = 30


@dc.dataclass
class MarxbotObservation(DifferentialDriveObservation):
    """
    Observation configuration for :py:class:`pyenki.Marxbot`.

    If ``normalize=True``, the scanner readings will be normalized in [0, 1]
    according to ``max_scanner_range``.

    Uses by default ``max_speed=30``.
    """
    max_speed: float = 30
    camera: bool = False
    """If set, observations will include field ``"camera"``
    with the readings from the omni-directional linear RGB camera.
    """
    scanner: bool = True
    """If set, observations will include field ``"scanner"``
    with the readings from the laser scanner.
    """
    max_scanner_range: float = 100
    """the maximal range of the scanner"""

    def get(self, r: DifferentialWheeled) -> Observation:
        robot = cast('Marxbot', r)
        ks = super().get(r)
        if self.camera:
            ks['camera'] = robot.scanner_image[:, :3].astype(self.dtype)
        if self.scanner:
            ks['scanner'] = np.clip(robot.scanner_distances.astype(self.dtype),
                                    0, self.max_scanner_range)
            if self.normalize:
                ks['scanner'] /= self.max_scanner_range
        return ks

    @property
    def space(self) -> gym.spaces.Dict:
        ks = super().space
        if self.camera:
            ks['camera'] = gym.spaces.Box(low=0,
                                          high=1,
                                          shape=(180, 3),
                                          dtype=self.dtype)
        if self.scanner:
            ks['scanner'] = gym.spaces.Box(
                low=0,
                high=1 if self.normalize else self.max_scanner_range,
                shape=(180, ),
                dtype=self.dtype)
        return ks


@dc.dataclass
class MarxbotConfig(GroupConfig):
    """
    The default configuration for a :py:class:`pyenki.Marxbot`
    that instantiates :py:class:`MarxbotAction` and :py:class:`MarxbotObservation`
    with their default parameters.
    """
    action: ActionConfig = dc.field(default_factory=MarxbotAction)
    observation: ObservationConfig = dc.field(
        default_factory=MarxbotObservation)


@dc.dataclass
class EPuckAction(DifferentialDriveAction):
    """
    Action configuration for :py:class:`pyenki.EPuck`.

    Uses by default ``max_speed=12.8``.
    """
    max_speed: float = 12.8


@dc.dataclass
class EPuckObservation(DifferentialDriveObservation):
    """
    Observation configuration for :py:class:`pyenki.EPuck`.

    If ``normalize=True``, fields ``scanner``, ``prox/distance``, and ``prox/value``
    will be normalized in [0, 1], according to ``max_scanner_range``,
    ``max_proximity_distance`` and ``max_proximity_value``.

    Uses by default ``max_speed=12.8``.
    """

    max_speed: float = 12.8
    camera: bool = False
    """
    If set, observations will include field ``"camera"``
    with readings from the linear RGB camera.
    """
    scanner: bool = False
    """
    If set, observations will include field ``"scanner"``
    with readings from the laser scanner.
    """
    proximity_distance: bool = False
    """
    If set, observations will include field ``"prox/distance"``
    with distances measured by the proximity sensors.
    """
    proximity_value: bool = True
    """
    If set, observations will include field ``"prox/value"``
    with readings from the proximity sensors.
    """
    max_scanner_range: float = 32
    """the maximal range of the scanner"""
    max_proximity_distance: float = 12
    """the maximal range of the proximity sensors"""
    max_proximity_value: float = 3731
    """the maximal values read by proximity sensors"""

    def get(self, r: DifferentialWheeled) -> Observation:
        robot = cast('EPuck', r)
        ks = super().get(r)
        if self.proximity_distance:
            ks['prox/distance'] = np.clip(
                robot.prox_distances.astype(self.dtype), 0,
                self.max_proximity_distance)
            if self.normalize:
                ks['prox/distance'] /= self.max_proximity_distance
        if self.proximity_value:
            ks['prox/value'] = np.clip(robot.prox_values.astype(self.dtype), 0,
                                       self.max_proximity_value)
            if self.normalize:
                ks['prox/value'] /= self.max_proximity_value
        if self.camera:
            ks['camera'] = robot.camera_image[:, :3].astype(self.dtype)
        if self.scanner:
            ks['scanner'] = np.clip(robot.scan.astype(self.dtype), 0,
                                    self.max_scanner_range)
            if self.normalize:
                ks['scanner'] /= self.max_scanner_range
        return ks

    @property
    def space(self) -> gym.spaces.Dict:
        ks = super().space
        if self.proximity_distance:
            ks['prox/distance'] = gym.spaces.Box(low=0,
                                                 high=1,
                                                 shape=(8, ),
                                                 dtype=self.dtype)
        if self.proximity_value:
            ks['prox/value'] = gym.spaces.Box(low=0,
                                              high=1,
                                              shape=(8, ),
                                              dtype=self.dtype)
        if self.camera:
            ks['camera'] = gym.spaces.Box(low=0,
                                          high=1,
                                          shape=(60, 3),
                                          dtype=self.dtype)
        if self.scanner:
            ks['scanner'] = gym.spaces.Box(
                low=0,
                high=1 if self.normalize else self.max_scanner_range,
                shape=(32, ),
                dtype=self.dtype)
        return ks


@dc.dataclass
class EPuckConfig(GroupConfig):
    """
    The default configuration for a :py:class:`pyenki.EPuck`
    that instantiates :py:class:`EPuckAction` and :py:class:`EPuckObservation`
    with their default parameters.
    """
    action: ActionConfig = dc.field(default_factory=EPuckAction)
    observation: ObservationConfig = dc.field(default_factory=EPuckObservation)


@dc.dataclass
class ThymioAction(DifferentialDriveAction):
    """
    Action configuration for :py:class:`pyenki.Thymio2`.
    Uses by default ``max_speed=16.6``.
    """
    max_speed: float = 16.6
    prox_comm: bool = False
    """
    If set, there is an additional action in [-1, 1]
    that is converted to an integer in ``[0, max_proximity_comm_payload]``
    before being broadcasted using proximity sensors.
    """
    max_proximity_comm_payload: int = 1
    """the maximal value of broadcasted messages"""

    def actuate(self, act: Action, robot: DifferentialWheeled,
                dt: float) -> None:
        super().actuate(act, robot, dt)
        if self.prox_comm:
            thymio = cast('Thymio2', robot)
            thymio.prox_comm_enabled = True
            thymio.prox_comm_tx = round(0.5 * (act[-1] + 1) *
                                        self.max_proximity_comm_payload)

    @property
    def space(self) -> gym.spaces.Box:
        s = super().space
        if self.prox_comm:
            return gym.spaces.Box(low=-1,
                                  high=1,
                                  shape=(s.shape[0] + 1, ),
                                  dtype=self.dtype)
        return s


@dc.dataclass
class ThymioObservation(DifferentialDriveObservation):
    """
    Observation configuration for :py:class:`pyenki.Thymio2`.

    If ``normalize=True``, fields ``prox/distance``, ``prox/value``,
    ``prox_comm/payload``, ``prox_comm/intensity``, and ``prox_comm/rx``
    will be normalized in [0, 1], according to the values of
    ``max_proximity_distance``, ``max_proximity_value``,
    ``max_proximity_comm_payload``, and ``max_proximity_comm_intensity``.

    ``proximity_comm_xxx`` fields are sorted by the intensity of the messages,
    from highest to lowest.

    Uses by default ``max_speed=12.8``.
    """
    max_speed: float = 16.6
    proximity_distance: bool = False
    """
    If set, observations will include field ``"prox/distance"``
    with distances measured by the proximity sensors.
    """
    proximity_value: bool = True
    """
    If set, observations will include field ``"prox/value"``
    with readings from the proximity sensors.
    """
    proximity_comm_payload: bool = False
    """
    If set, observations will include field ``"prox_comm/payload"``
    with the payload of IR messages received by proximity sensors.
    """
    proximity_comm_intensity: bool = False
    """
    If set, observations will include field ``"prox_comm/intensity"``
    with the intensity of IR messages received by proximity sensors.
    """
    proximity_comm_rx: bool = False
    """
    If set, observations will include field ``"prox_comm/rx"``
    with IR messages.
    """
    max_proximity_distance: float = 28
    """the maximal range of the proximity sensors"""
    max_proximity_value: float = 4505
    """the maximal values read by proximity sensors"""
    max_proximity_comm_payload: int = 1
    """the maximal proximity communication payload"""
    max_proximity_comm_intensity: float = 4600
    """the maximal proximity communication intensity"""
    max_proximity_comm_number: int = 1
    """
    the maximal number of proximity comm messages.
    If more messages are received, messages with the
    lowest intensity are ignored.
    """

    def get_prox_comm_events(self,
                             thymio: Thymio2) -> list[Thymio2.IRCommEvent]:
        u = any((self.proximity_distance, self.proximity_value,
                 self.proximity_comm_intensity, self.proximity_comm_rx))
        if not u or self.max_proximity_comm_number <= 0:
            return []
        events = thymio.prox_comm_events
        events = sorted(events, key=lambda e: max(e.intensities), reverse=True)
        return events[:self.max_proximity_comm_number]

    def get(self, r: DifferentialWheeled) -> Observation:
        robot = cast('Thymio2', r)
        ks = super().get(r)
        if self.proximity_distance:
            ks['prox/distance'] = np.clip(
                robot.prox_distances.astype(self.dtype), 0,
                self.max_proximity_distance)
            if self.normalize:
                ks['prox/distance'] /= self.max_proximity_distance

        if self.proximity_value:
            ks['prox/value'] = np.clip(robot.prox_values.astype(self.dtype), 0,
                                       self.max_proximity_value)
            if self.normalize:
                ks['prox/value'] /= self.max_proximity_value
        events = self.get_prox_comm_events(robot)
        if self.proximity_comm_payload:
            ps = [e.payloads for e in events]
            if ps:
                vs = np.array(ps, dtype=int)
            else:
                vs = np.empty((self.max_proximity_comm_number, 7), dtype=int)
            if len(vs) < self.max_proximity_comm_number:
                m = self.max_proximity_comm_number - len(vs)
                vs = np.concatenate([vs, np.zeros((m, 7), dtype=int)])
            ks['prox_comm/payload'] = np.clip(vs, 0,
                                              self.max_proximity_comm_payload)
            if self.normalize:
                ks['prox_comm/payload'] /= self.max_proximity_comm_payload
        if self.proximity_comm_intensity:
            rs = [e.intensities for e in events]
            if rs:
                vs = np.asarray(rs, dtype=self.dtype)
            else:
                vs = np.zeros((self.max_proximity_comm_number, 7),
                              dtype=self.dtype)
            if len(vs) < self.max_proximity_comm_number:
                m = self.max_proximity_comm_number - len(vs)
                vs = np.concatenate([vs, np.zeros((m, 7))])
            ks['prox_comm/intensity'] = np.clip(
                vs, 0, self.max_proximity_comm_intensity)
            if self.normalize:
                ks['prox_comm/intensity'] /= float(
                    self.max_proximity_comm_intensity)
        if self.proximity_comm_rx:
            qs = [e.rx_value for e in events]
            if qs:
                vs = np.array(qs, dtype=int)
            else:
                vs = np.zeros(self.max_proximity_comm_number, dtype=int)
            if len(vs) < self.max_proximity_comm_number:
                m = self.max_proximity_comm_number - len(vs)
                vs = np.concatenate([vs, np.zeros(m, dtype=int)])

            ks['prox_comm/rx'] = np.clip(vs, 0,
                                         self.max_proximity_comm_payload)
            if self.normalize:
                ks['prox_comm/rx'] /= self.max_proximity_comm_payload
        return ks

    @property
    def space(self) -> gym.spaces.Dict:
        ks = super().space
        if self.proximity_distance:
            ks['prox/distance'] = gym.spaces.Box(
                low=0,
                high=1 if self.normalize else self.max_proximity_distance,
                shape=(7, ),
                dtype=self.dtype)
        if self.proximity_value:
            ks['prox/value'] = gym.spaces.Box(
                low=0,
                high=1 if self.normalize else self.max_proximity_value,
                shape=(7, ),
                dtype=self.dtype)
        if self.proximity_comm_payload:
            ks['prox_comm/payload'] = gym.spaces.Box(
                low=0,
                high=1 if self.normalize else self.max_proximity_comm_payload,
                shape=(self.max_proximity_comm_number, 7),
                dtype=np.int64)
        if self.proximity_comm_intensity:
            ks['prox_comm/intensity'] = gym.spaces.Box(
                low=0,
                high=1
                if self.normalize else self.max_proximity_comm_intensity,
                shape=(self.max_proximity_comm_number, 7),
                dtype=self.dtype)
        if self.proximity_comm_rx:
            ks['prox_comm/rx'] = gym.spaces.Box(
                low=0,
                high=1 if self.normalize else self.max_proximity_comm_payload,
                shape=(self.max_proximity_comm_number, ),
                dtype=np.int64)
        return ks


@dc.dataclass
class ThymioConfig(GroupConfig):
    """
    The default configuration for a :py:class:`pyenki.Thymio2`
    that instantiates :py:class:`ThymioAction` and :py:class:`ThymioObservation`
    with their default parameters.
    """
    action: ActionConfig = dc.field(default_factory=ThymioAction)
    observation: ObservationConfig = dc.field(
        default_factory=ThymioObservation)
