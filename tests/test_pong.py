from pathlib import Path

import pytest

from pong import Pong, PongReward
from fastconnectome.models.malecns import Turn


def test_tracking_reward_reports_horizontal_alignment() -> None:
    game = Pong(reward=PongReward.TRACKING, tracking_tolerance=10)
    game.reset()
    game.state.ball_vx = 0
    game.state.ball_vy = 0
    game.state.ball_x = game.state.paddle_x + game.paddle_width / 2

    aligned = game.step(Turn.HOLD)
    game.state.ball_x = 0
    unaligned = game.step(Turn.HOLD)

    assert aligned.reward == 1
    assert aligned.info["aligned"] is True
    assert unaligned.reward == -1
    assert unaligned.info["aligned"] is False


def test_tracking_configuration_is_checkpointed(tmp_path: Path) -> None:
    source = Pong(reward=PongReward.TRACKING, tracking_tolerance=12)
    source.reset()
    checkpoint = tmp_path / "pong.npz"
    source.save(checkpoint)

    matching = Pong(reward=PongReward.TRACKING, tracking_tolerance=12)
    matching.restore(checkpoint)

    mismatched = Pong(reward=PongReward.SPARSE, tracking_tolerance=12)
    with pytest.raises(ValueError, match="configuration"):
        mismatched.restore(checkpoint)
