import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GameScoreCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._scores = []
        self._lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done and "score" in info:
                self._scores.append(info["score"])
                if "episode" in info:
                    self._lengths.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self) -> None:
        if self._scores:
            self.logger.record("rollout/mean_game_score", np.mean(self._scores))
        if self._scores and self._lengths:
            ratios = [s / l for s, l in zip(self._scores, self._lengths)]
            self.logger.record("rollout/mean_score_per_step", np.mean(ratios))
        self._scores = []
        self._lengths = []
