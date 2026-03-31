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
                    # Boltzmann: sample ∝ exp(Q / τ)
                    valid = pa > 0
                    logits = np.full_like(p, -np.inf)
                    logits[valid] = p[valid] / self.tau
                    logits[valid] -= logits[valid].max()   # log-sum-exp stability
                    probs = np.zeros_like(p)
                    probs[valid] = np.exp(logits[valid])
                    probs /= probs.sum()

                    ai = np.random.choice(len(self.ACTIONS), p=probs)
                    actions[sy] = {
                        "action_private": self.ACTIONS[ai],
                        "method": "boltzmann",
                        "actions_possible": pa,
                        "predictions": p,
                        "action_index": ai,
                    }

        for sy in symbols:
            actions[sy] = self._shape_action(actions[sy], sy, state)

        return actions
