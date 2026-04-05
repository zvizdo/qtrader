"""Boltzmann (softmax) exploration variant of DQTPAgent.

Instead of ε-greedy (random coin flip), actions are sampled proportional
to exp(Q / τ).  The parent's expl_rate/min/decay are reused as the
temperature schedule — ready_to_learn(), load_config(), save_config()
and all other logic inherit unchanged.
"""

import numpy as np
from typing import Dict
from qtrader.agents.dqtp import DQTPAgent


class BoltzmannDQTPAgent(DQTPAgent):

    def __init__(self, *args, boltz_uniform_floor: float = 0.05, **kwargs):
        """Boltzmann exploration agent.

        Parameters
        ----------
        boltz_uniform_floor : float
            Mixing weight for a uniform distribution over valid actions,
            added to the softmax probabilities to guarantee a minimum
            exploration rate even when τ decays. Set to 0 to disable.
            Default 0.05 (5% of probability mass on uniform).
        """
        super().__init__(*args, **kwargs)
        self.boltz_uniform_floor = float(boltz_uniform_floor)

    # ── semantic alias ──────────────────────────────────────────────

    @property
    def tau(self) -> float:
        """Temperature (aliases expl_rate for clarity)."""
        return self.expl_rate

    @tau.setter
    def tau(self, value: float):
        self.expl_rate = value

    # ── action selection (the only method that changes) ─────────────

    def act(self, state: dict) -> Dict:
        symbols = state["state_global"]["symbols"]

        actions = {}
        for sy in symbols:
            pa = self._possible_actions(sy, state)

            if self.model_online is None:
                # No model yet — uniform random (same as parent)
                exploration_bias = np.array([0.5, 0.5]) * pa
                exploration_bias /= exploration_bias.sum()
                ai = np.random.choice(len(self.ACTIONS), p=exploration_bias)
                actions[sy] = {
                    "action_private": self.ACTIONS[ai],
                    "method": "random",
                    "actions_possible": pa,
                    "action_index": ai,
                }

            else:
                ex = self._generate_example(sy, state)
                ex = np.array([ex], dtype=np.float64)
                p = self.model_online.predict(ex, verbose=0)[0]

                if self.no_learn:
                    # Eval / live: greedy (identical to parent)
                    p[pa == 0] = -np.inf
                    ai = np.argmax(p)
                    actions[sy] = {
                        "action_private": self.ACTIONS[ai],
                        "method": "model",
                        "actions_possible": pa,
                        "predictions": p,
                        "action_index": ai,
                    }

                else:
                    # Boltzmann: sample ∝ exp(Q / τ), mixed with a uniform
                    # floor over valid actions to guarantee exploration
                    # survives τ decay. probs = (1-f)*softmax + f*uniform.
                    valid = pa > 0
                    logits = np.full_like(p, -np.inf)
                    logits[valid] = p[valid] / self.tau
                    logits[valid] -= logits[valid].max()   # log-sum-exp stability
                    softmax_probs = np.zeros_like(p)
                    softmax_probs[valid] = np.exp(logits[valid])
                    softmax_probs /= softmax_probs.sum()

                    f = self.boltz_uniform_floor
                    if f > 0.0:
                        n_valid = int(valid.sum())
                        uniform = np.zeros_like(p)
                        uniform[valid] = 1.0 / n_valid
                        probs = (1.0 - f) * softmax_probs + f * uniform
                        probs /= probs.sum()
                    else:
                        probs = softmax_probs

                    ai = np.random.choice(len(self.ACTIONS), p=probs)

                    # Argmax-rate diagnostic: how often the sampled action
                    # equals the greedy argmax (1.0 = fully deterministic).
                    masked = np.where(valid, p, -np.inf)
                    argmax_ai = int(np.argmax(masked))
                    self.boltz_argmax_count += int(ai == argmax_ai)
                    self.boltz_sample_count += 1

                    actions[sy] = {
                        "action_private": self.ACTIONS[ai],
                        "method": "boltzmann",
                        "actions_possible": pa,
                        "predictions": p,
                        "action_index": ai,
                    }

        for sy in symbols:
            actions[sy] = self._shape_action(actions[sy], sy, state)

        # -- Action-fraction tracking (mirrors parent DQTPAgent.act) --
        for sy in symbols:
            ap = actions[sy].get("action_private")
            if ap == self.ACTION_FLAT:
                self.action_flat_count += 1
            elif ap == self.ACTION_LONG:
                self.action_long_count += 1
            self.action_total += 1

        return actions
