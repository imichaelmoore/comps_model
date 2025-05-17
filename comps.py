#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mesa<3",
#   "numpy>=1.25",
#   "pandas>=2.1",
#   "matplotlib>=3.8"
# ]
# ///

"""Agent-based model of hierarchical information aggregation.

This file contains a **pure-Python re-implementation** of the model described in
Michael A. Moore's comps paper *"Bayesian Decision Making in Intelligence
Bureaucracies."*  It reproduces the continuous-space variant of Moore's original
NetLogo simulation with the **Mesa** framework, letting you explore two simple
aggregation strategies:

* **bayesian** - field officers average their previous estimate with a new noisy
  sample (recursive Bayes).
* **nonlearning** - each level takes the plain arithmetic mean of the most
  recent messages.

Run ``python comps.py --gui`` for an interactive web dashboard, or omit
``--gui`` for a head-less simulation that prints summary metrics.

Generated in cooperation with **OpenAI o3** in May 2025.
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation

# ---------------------------------------------------------------------------#
# Constants                                                                  #
# ---------------------------------------------------------------------------#
GRID_RES = 41  # number of cells per side (≈ 0.5 NetLogo patch)
WORLD_MIN = -10
WORLD_MAX = 10
CELL_SCALE = (GRID_RES - 1) / (WORLD_MAX - WORLD_MIN)  # 2.0 for 41x41

# ---------------------------------------------------------------------------#
# Agents                                                                     #
# ---------------------------------------------------------------------------#


class TruthMarker(Agent):
    """Passive marker placed at the ground-truth coordinates ``(H₀, H₁)``."""

    def __init__(
        self,
        uid: int | str,
        model: OrgModel,
        x: float,
        y: float,
    ) -> None:
        """Create an immutable marker located at ``(x, y)``."""
        super().__init__(uid, model)
        self.x: float = x
        self.y: float = y


class OrgAgent(Agent):
    """Generic organisational actor (*agent*, *analyst*, *manager*, *director*)."""

    def __init__(
        self,
        uid: int,
        model: OrgModel,
        rank: str,
        party: str,
        supervisor: OrgAgent | None = None,
    ) -> None:
        """Initialise an organisational node in the hierarchy."""
        super().__init__(uid, model)
        self.rank: str = rank
        self.party: str = party
        self.supervisor: OrgAgent | None = supervisor

        self.ph_zero, self.ph_one = self._initial_belief()
        self._next: tuple[float, float] | None = None

    # ---------------------------------------------------------------------#
    # Internal helpers                                                      #
    # ---------------------------------------------------------------------#
    def _initial_belief(self) -> tuple[float, float]:
        """Return an initial noisy observation of the ground truth."""
        h0, h1 = self.model.truth
        sd = self.model.reliability
        rng = self.model.rng
        return rng.normal(h0, sd), rng.normal(h1, sd)

    def _biased_sample(self) -> tuple[float, float]:
        """Take a fresh sample; invert it when the party is marked *wrong*."""
        h0, h1 = self.model.truth
        sd = self.model.reliability
        rng = self.model.rng
        x, y = rng.normal(h0, sd), rng.normal(h1, sd)

        if self.model.parties_status == "blue wrong" and self.party == "blue":
            x, y = -x, -y
        if self.model.parties_status == "red wrong" and self.party == "red":
            x, y = -x, -y
        return x, y

    # ---------------------------------------------------------------------#
    # Mesa API                                                              #
    # ---------------------------------------------------------------------#
    def step(self) -> None:
        """Compute the next belief but *do not* commit it."""
        if self.rank == "agent":
            x_new, y_new = self._biased_sample()
            if self.model.algo == "bayesian":
                x_new = (self.ph_zero + x_new) / 2
                y_new = (self.ph_one + y_new) / 2
            self._next = (x_new, y_new)
        else:
            subs = self.model.subordinates[self]
            if self.model.same_party_filter:
                subs = [a for a in subs if a.party == self.party]
            xs = [a.ph_zero for a in subs] + [self.ph_zero]
            ys = [a.ph_one for a in subs] + [self.ph_one]
            self._next = (float(np.mean(xs)), float(np.mean(ys)))

    def advance(self) -> None:
        """Commit the pending belief and move the sprite to the grid cell."""
        x, y = self._next  # type: ignore[assignment]
        x = max(WORLD_MIN, min(WORLD_MAX, x))
        y = max(WORLD_MIN, min(WORLD_MAX, y))
        self.ph_zero, self.ph_one = x, y

        cell = self.model._world_to_cell(x, y)
        self.model.grid.move_agent(self, cell)


# ---------------------------------------------------------------------------#
# Model                                                                      #
# ---------------------------------------------------------------------------#


class OrgModel(Model):
    """Two-dimensional continuous opinion-dynamics model of a four-tier hierarchy."""

    def __init__(
        self,
        reliability: float = 1.5,
        party_ratio: float = 0.5,
        algo: str = "bayesian",
        parties_status: str = "neutral",
        same_party_filter: bool = False,
        eq_threshold: float = 0.01,
        n_managers: int = 3,
        n_analysts: int = 4,
        n_agents: int = 15,
        seed: int | None = None,
    ) -> None:
        """Configure and build a new simulation instance."""
        super().__init__(seed=seed)

        # Simulation switches
        self.reliability: float = reliability
        self.party_ratio: float = party_ratio
        self.algo: str = algo
        self.parties_status: str = parties_status
        self.same_party_filter: bool = same_party_filter
        self.eq_threshold: float = eq_threshold

        # RNG & ground truth
        self.rng = np.random.default_rng(seed)
        self.truth: tuple[float, float] = tuple(
            self.random.uniform(WORLD_MIN, WORLD_MAX) for _ in range(2)
        )

        # Mesa engine objects
        self.grid: MultiGrid = MultiGrid(GRID_RES, GRID_RES, torus=False)
        self.schedule: SimultaneousActivation = SimultaneousActivation(self)
        self.subordinates: dict[OrgAgent, list[OrgAgent]] = {}
        self.cycles_since_move: int = 0

        self._build_hierarchy(n_managers, n_analysts, n_agents)

        # Visible truth marker
        star = TruthMarker("truth", self, *self.truth)
        self._place(star)

        # Data collector
        self.datacollector: DataCollector = DataCollector(
            model_reporters={
                "director_error": self.director_error,
                "tick": lambda m: m.schedule.time,
            },
        )

    # ---------------------------------------------------------------------#
    # Internal helpers                                                     #
    # ---------------------------------------------------------------------#
    def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Convert continuous coordinates to integer grid indices."""
        cx = round((x - WORLD_MIN) * CELL_SCALE)
        cy = round((y - WORLD_MIN) * CELL_SCALE)
        cx = max(0, min(self.grid.width - 1, cx))
        cy = max(0, min(self.grid.height - 1, cy))
        return cx, cy

    def _place(self, agent: Agent) -> None:
        """Register *agent* with the scheduler and place it on the grid."""
        self.schedule.add(agent)
        pos = (
            self._world_to_cell(agent.x, agent.y)  # type: ignore[attr-defined]
            if isinstance(agent, TruthMarker)
            else self._world_to_cell(  # type: ignore[attr-defined]
                agent.ph_zero, agent.ph_one
            )
        )
        self.grid.place_agent(agent, pos)

    def _rand_party(self) -> str:
        """Return "blue" with probability *party_ratio*, otherwise "red"."""
        return "blue" if self.random.random() < self.party_ratio else "red"

    def _build_hierarchy(
        self,
        n_mgr: int,
        n_anal: int,
        n_ag: int,
    ) -> None:
        """Instantiate the full organisational tree."""
        uid_counter = itertools.count()

        director = OrgAgent(next(uid_counter), self, "director", "red")
        self._place(director)

        managers: list[OrgAgent] = []
        for _ in range(n_mgr):
            mgr = OrgAgent(next(uid_counter), self, "manager", "red", director)
            self._place(mgr)
            managers.append(mgr)
        self.subordinates[director] = managers

        analysts: list[OrgAgent] = []
        for mgr in managers:
            sub: list[OrgAgent] = []
            for _ in range(n_anal):
                anal = OrgAgent(next(uid_counter), self, "analyst", "red", mgr)
                self._place(anal)
                sub.append(anal)
                analysts.append(anal)
            self.subordinates[mgr] = sub

        for anal in analysts:
            sub: list[OrgAgent] = []
            for _ in range(n_ag):
                ag = OrgAgent(next(uid_counter), self, "agent", self._rand_party(), anal)
                self._place(ag)
                sub.append(ag)
            self.subordinates[anal] = sub

    # ---------------------------------------------------------------------#
    # Metrics                                                              #
    # ---------------------------------------------------------------------#
    def director_error(self) -> float:
        """Return the Euclidean distance between the director's belief and truth."""
        director = next(a for a in self.schedule.agents if getattr(a, "rank", "") == "director")
        dx = director.ph_zero - self.truth[0]
        dy = director.ph_one - self.truth[1]
        return float(np.hypot(dx, dy))

    # ---------------------------------------------------------------------#
    # Main tick                                                            #
    # ---------------------------------------------------------------------#
    def step(self) -> None:
        """Advance the simulation by one tick and halt on equilibrium."""
        director = next(a for a in self.schedule.agents if getattr(a, "rank", "") == "director")
        prev = (director.ph_zero, director.ph_one)

        self.schedule.step()

        move = np.hypot(director.ph_zero - prev[0], director.ph_one - prev[1])
        self.cycles_since_move = self.cycles_since_move + 1 if move < self.eq_threshold else 0
        self.datacollector.collect(self)

        if self.cycles_since_move >= 10:
            self.running = False


# ---------------------------------------------------------------------------#
# Visualisation                                                              #
# ---------------------------------------------------------------------------#
try:
    from mesa.visualization.ModularVisualization import ModularServer
    from mesa.visualization.modules import CanvasGrid, ChartModule
except ImportError:  # pragma: no cover — GUI is optional
    ModularServer = CanvasGrid = ChartModule = None  # type: ignore[assignment]


def agent_portrayal(agent: Agent) -> dict[str, object]:
    """Return a serialisable description for Mesa's JavaScript frontend."""
    # ------------------------- truth marker ------------------------------#
    if isinstance(agent, TruthMarker):
        return {
            "Shape": "star",
            "x": (agent.x - WORLD_MIN) * CELL_SCALE,
            "y": (agent.y - WORLD_MIN) * CELL_SCALE,
            "r": 2,
            "Color": "#ffe066",
            "Filled": "true",
            "Layer": 2,
        }

    # ------------------------ organisational ranks -----------------------#
    colour_map = {
        "agent": "#8b4513",
        "analyst": "#ff69b4",
        "manager": "#32cd32",
        "director": "#ffd700",
    }
    radius_map = {"agent": 1.3, "analyst": 1.6, "manager": 2.2, "director": 3.0}

    return {
        "Shape": "circle",
        "x": (agent.ph_zero - WORLD_MIN) * CELL_SCALE,
        "y": (agent.ph_one - WORLD_MIN) * CELL_SCALE,
        "r": radius_map[agent.rank],
        "Color": colour_map[agent.rank],
        "Filled": "true",
        "Layer": 1,
    }


# ---------------------------------------------------------------------------#
# Command-line interface                                                     #
# ---------------------------------------------------------------------------#
def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run either the GUI or a head-less batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="launch the interactive viewer")
    parser.add_argument(
        "--algo",
        choices=["bayesian", "nonlearning"],
        default="bayesian",
        help="aggregation algorithm (default: bayesian)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="benchmark both algorithms for several noise levels",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="number of random seeds to average in batch mode",
    )
    args = parser.parse_args(argv)

    if args.gui:
        if ModularServer is None:
            sys.exit("Mesa GUI dependencies are not installed")
        grid = CanvasGrid(agent_portrayal, GRID_RES, GRID_RES, 600, 600)
        chart = ChartModule([{"Label": "director_error", "Color": "black"}])
        ModularServer(
            OrgModel,
            [grid, chart],
            "Hierarchy - continuous view",
            {"algo": args.algo},
        ).launch()
        return

    # ------------------------------ helpers ------------------------------#
    def run_once(seed: int = 0) -> tuple[int, float]:
        """Run a single simulation and return ``(ticks, final_director_error)``."""
        model = OrgModel(algo=args.algo, seed=seed)
        ticks = 0
        while model.running and ticks < 2_000:
            model.step()
            ticks += 1
        return ticks, model.director_error()

    # -------------------------- batch or single --------------------------#
    if args.batch:
        records: list[dict[str, object]] = []
        for algo, rel in itertools.product(
            ["bayesian", "nonlearning"],
            [0.5, 1.5, 3.0],
        ):
            times: list[int] = [run_once(seed=s)[0] for s in range(args.runs)]
            records.append({"algo": algo, "reliability": rel, "mean_ticks": float(np.mean(times))})

        df = pd.DataFrame(records)
        print(df.to_string(index=False))
        return

    ticks, err = run_once()
    print(f"finished in {ticks} ticks; director final error {err:.3f}")


if __name__ == "__main__":
    main()
