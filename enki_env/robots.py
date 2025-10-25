from __future__ import annotations

import dataclasses as dc
from typing import cast

import gymnasium as gym
import numpy as np
from pyenki import DifferentialWheeled, EPuck, Marxbot, Thymio2

from .config import ActionConfig, ObservationConfig
from .types import Action, Observation


@dc.dataclass
class DifferentialDriveAction(ActionConfig):

    max_speed: float = 0
    acceleration: bool = False
    max_acceleration: float = 1

    def actuate(self, act: Action, robot: DifferentialWheeled,
                dt: float) -> None:
        if self.acceleration:
            s = robot.left_wheel_encoder_speed + act[
                0] * self.max_acceleration * dt
            robot.left_wheel_target_speed = min(self.max_speed,
                                                max(s, -self.max_speed))
            s = robot.right_wheel_encoder_speed + act[
                1] * self.max_acceleration * dt
            robot.right_wheel_target_speed = min(self.max_speed,
                                                 max(s, -self.max_speed))
        else:
            robot.left_wheel_target_speed = act[0] * self.max_speed
            robot.right_wheel_target_speed = act[1] * self.max_speed

    @property
    def space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1, high=1, shape=(2, ))


@dc.dataclass
class DifferentialDriveObservation(ObservationConfig):
    max_speed: float = 0
    speed: bool = False

    def get(self, robot: DifferentialWheeled) -> Observation:
        ks = {}
        if self.speed:
            ks['wheel_speeds'] = np.array([
                robot.left_wheel_encoder_speed / self.max_speed,
                robot.right_wheel_encoder_speed / self.max_speed
            ])
        return ks

    @property
    def space(self) -> gym.spaces.Dict:
        ks: dict[str, gym.Space] = {}
        if self.speed:
            ks['wheel_speeds'] = gym.spaces.Box(low=-1,
                                                high=1,
                                                shape=(2, ),
                                                dtype=np.float64)
        return gym.spaces.Dict(ks)


@dc.dataclass
class MarxbotAction(DifferentialDriveAction):
    max_speed: float = 30


@dc.dataclass
class MarxbotObservation(DifferentialDriveObservation):
    max_speed: float = 30
    camera: bool = False
    lidar: bool = True
    normalize: bool = True
    max_range: float = 100

    def get(self, r: DifferentialWheeled) -> Observation:
        robot = cast(Marxbot, r)
        ks = {}
        if self.camera:
            ks['camera'] = robot.scanner_image[:, :3]
        if self.lidar:
            ks['lidar'] = np.clip(robot.scanner_distances / self.max_range, 0,
                                  1)
        return ks

    @property
    def space(self) -> gym.spaces.Dict:
        ks: dict[str, gym.Space] = {}
        if self.camera:
            ks['camera'] = gym.spaces.Box(low=0,
                                          high=1,
                                          shape=(180, 3),
                                          dtype=np.float64)
        if self.lidar:
            ks['lidar'] = gym.spaces.Box(low=0,
                                         high=1,
                                         shape=(180, ),
                                         dtype=np.float64)
        return gym.spaces.Dict(ks)


@dc.dataclass
class EPuckAction(DifferentialDriveAction):
    max_speed: float = 12.8


@dc.dataclass
class EPuckObservation(DifferentialDriveObservation):
    max_speed: float = 12.8
    camera: bool = True
    lidar: bool = False
    proximity_distance: bool = False
    proximity_value: bool = True
    normalize: bool = True
    max_lidar_range: float = 100
    max_proximity_distance: float = 12
    max_proximity_value: float = 3731

    def get(self, r: DifferentialWheeled) -> Observation:
        robot = cast(EPuck, r)
        ks = {}
        if self.proximity_distance:
            ks['prox/distance'] = robot.prox_values / self.max_proximity_value
        if self.proximity_value:
            ks['prox/value'] = robot.prox_distances / self.max_proximity_distance
        if self.camera:
            ks['camera'] = robot.camera_image[:, :3]
        # if self.lidar:
        #     ks['lidar'] = np.clip(robot.scanner_distances / self.max_range, 0,
        #                           1)
        return ks

    @property
    def space(self) -> gym.spaces.Dict:
        ks: dict[str, gym.Space] = {}
        if self.proximity_distance:
            ks['prox/distance'] = gym.spaces.Box(low=0,
                                                 high=1,
                                                 shape=(8, ),
                                                 dtype=np.float64)
        if self.proximity_value:
            ks['prox/value'] = gym.spaces.Box(low=0,
                                              high=1,
                                              shape=(8, ),
                                              dtype=np.float64)
        if self.camera:
            ks['camera'] = gym.spaces.Box(low=0,
                                          high=1,
                                          shape=(60, 3),
                                          dtype=np.float64)
        # if self.lidar:
        #     ks['lidar'] = gym.spaces.Box(low=0,
        #                                  high=1,
        #                                  shape=(32, ),
        #                                  dtype=np.float64)
        return gym.spaces.Dict(ks)


@dc.dataclass
class ThymioAction(DifferentialDriveAction):
    max_speed: float = 16.6


@dc.dataclass
class ThymioObservation(DifferentialDriveObservation):
    max_speed: float = 16.6
    proximity_comm_payload: bool = False
    proximity_comm_intensity: bool = False
    proximity_comm_rx: bool = False
    max_proximity_comm_number: int = 1
    proximity_distance: bool = False
    proximity_value: bool = True
    normalize: bool = True
    max_proximity_distance: float = 28
    max_proximity_value: float = 4505
    max_proximity_comm_intensities: float = 4600

    def get(self, r: DifferentialWheeled) -> Observation:
        robot = cast(Thymio2, r)
        ks = {}
        if self.proximity_distance:
            ks['prox/distance'] = robot.prox_values / self.max_proximity_value
        if self.proximity_value:
            ks['prox/value'] = robot.prox_distances / self.max_proximity_distance
        if self.proximity_comm_payload:
            ps = [
                e.payloads for e in
                robot.prox_comm_events[:self.max_proximity_comm_number]
            ]
            if ps:
                vs = np.array(ps, dtype=int)
            else:
                vs = np.empty((self.max_proximity_comm_number, 7), dtype=int)
            if len(vs) < self.max_proximity_comm_number:
                m = self.max_proximity_comm_number - len(vs)
                vs = np.concatenate([vs, np.zeros((m, 7), dtype=int)])
            ks['prox_comm/payload'] = vs
        if self.proximity_comm_intensity:
            ps = [
                e.intensities for e in
                robot.prox_comm_events[:self.max_proximity_comm_number]
            ]
            if ps:
                vs = np.array(ps)
            else:
                vs = np.empty((self.max_proximity_comm_number, 7))
            if len(vs) < self.max_proximity_comm_number:
                m = self.max_proximity_comm_number - len(vs)
                vs = np.concatenate([vs, np.zeros((m, 7))])
            ks['prox_comm/intensity'] = vs / self.max_proximity_comm_intensities
        if self.proximity_comm_rx:
            qs = [
                e.rx_value for e in
                robot.prox_comm_events[:self.max_proximity_comm_number]
            ]
            if qs:
                vs = np.array(qs, dtype=int)
            else:
                vs = np.zeros(self.max_proximity_comm_number, dtype=int)
            if len(vs) < self.max_proximity_comm_number:
                m = self.max_proximity_comm_number - len(vs)
                vs = np.concatenate([vs, np.zeros(m, dtype=int)])
            ks['prox_comm/rx'] = vs
        return ks

    @property
    def space(self) -> gym.spaces.Dict:
        ks: dict[str, gym.Space] = {}
        if self.proximity_distance:
            ks['prox/distance'] = gym.spaces.Box(low=0,
                                                 high=1,
                                                 shape=(7, ),
                                                 dtype=np.float64)
        if self.proximity_value:
            ks['prox/value'] = gym.spaces.Box(low=0,
                                              high=1,
                                              shape=(7, ),
                                              dtype=np.float64)
        if self.proximity_comm_payload:
            ks['prox_comm/payload'] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.max_proximity_comm_number, 7),
                dtype=np.int64)
        if self.proximity_comm_intensity:
            ks['prox_comm/intensity'] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.max_proximity_comm_number, 7),
                dtype=np.float64)
        if self.proximity_comm_rx:
            ks['prox_comm/rx'] = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.max_proximity_comm_number, ),
                dtype=np.int64)
        return gym.spaces.Dict(ks)
