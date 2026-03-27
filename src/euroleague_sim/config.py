from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Optional


@dataclass(frozen=True)
class EloConfig:
    base: float = 1500.0
    k: float = 20.0
    home_advantage: float = 65.0       # Elo points added to home team pre-game
    blend_recent: float = 0.65         # weight for most recent past season
    blend_older: float = 0.35          # weight for the older past season


@dataclass(frozen=True)
class MCConfig:
    """Monte Carlo parameters for margin simulation.

    mu = alpha1*A + alpha2*(EloDiff/25) + alpha3
    margin ~ Normal(mu, sigma)
    """
    alpha1: float = 0.9    # weight on net rating differential A
    alpha2: float = 1.0    # weight on scaled Elo differential
    alpha3: float = 1.5    # home-court edge in points
    sigma: float = 11.5    # std-dev of margin distribution (default)
    n_sims: int = 20_000   # simulations per game


@dataclass(frozen=True)
class ShrinkageConfig:
    """Early-season shrinkage: NetRtg_adj = (n/(n+k))*NetRtg + (k/(n+k))*0."""
    k_games: int = 6       # shrinkage strength (5-8 per spec)


@dataclass(frozen=True)
class MLConfig:
    """Configuration for ML model training (Logistic Regression + Ridge)."""
    logreg_C: float = 1.0            # inverse regularisation strength
    logreg_max_iter: int = 1000
    ridge_alpha: float = 1.0         # L2 regularisation strength
    model_dir: str = "models"        # directory for persisted model artefacts
    cv_folds: int = 5


@dataclass(frozen=True)
class SeasonConfig:
    competition: str = "E"       # 'E' Euroleague, 'U' Eurocup
    season_start_year: int = 2025  # current season (2025-26)
    history_seasons: int = 2       # past seasons for Elo prior


@dataclass(frozen=True)
class ProjectConfig:
    season: SeasonConfig = SeasonConfig()
    elo: EloConfig = EloConfig()
    mc: MCConfig = MCConfig()
    shrinkage: ShrinkageConfig = ShrinkageConfig()
    ml: MLConfig = MLConfig()

    @staticmethod
    def default() -> "ProjectConfig":
        return ProjectConfig()

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def load(path: Path) -> "ProjectConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        season = SeasonConfig(**data.get("season", {}))
        elo = EloConfig(**data.get("elo", {}))
        mc = MCConfig(**data.get("mc", {}))
        shrinkage = ShrinkageConfig(**data.get("shrinkage", {}))
        ml = MLConfig(**data.get("ml", {}))
        return ProjectConfig(
            season=season, elo=elo,
            mc=mc, shrinkage=shrinkage, ml=ml,
        )

    def dump(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
