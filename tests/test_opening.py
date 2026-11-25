import json

import numpy as np
import pytest

from ms.engine import BEGINNER, EXPERT, Config, Minefield
from ms.eval import main, run
from ms.opening import OPENINGS, first_cell


def test_random_is_reproducible_and_uses_more_than_one_cell():
    seq = [first_cell(EXPERT, seed, 'random') for seed in range(200)]
    assert seq == [first_cell(EXPERT, seed, 'random') for seed in range(200)]
    assert len(set(seq)) > 100
    assert any(divmod(i, EXPERT.cols)[0] == 0 for i in seq)
    assert any(divmod(i, EXPERT.cols)[0] == EXPERT.rows // 2 for i in seq)


@pytest.mark.parametrize('opening', OPENINGS)
def test_opening_geometry_and_rule(opening):
    for seed in range(40):
        idx = first_cell(EXPERT, seed, opening)
        row, col = divmod(idx, EXPERT.cols)
        assert 0 <= row < EXPERT.rows and 0 <= col < EXPERT.cols
        if opening == 'center':
            assert (row, col) == (8, 15)
        elif opening == 'interior':
            assert 0 < row < 15 and 0 < col < 29
        elif opening in ('near-corner', 'near-corner-3'):
            expected = 2 if opening == 'near-corner' else 3
            assert min(row, 15 - row) == min(col, 29 - col) == expected
        elif opening == 'edge':
            assert row in (0, 15) or col in (0, 29)
        game = Minefield(EXPERT, np.random.default_rng(seed))
        assert game.click(idx) != 'boom'
        assert game.visible[idx] == 0 and int(game.mines.sum()) == 99


@pytest.mark.parametrize('opening', OPENINGS)
def test_thin_board_does_not_have_an_empty_opening_pool(opening):
    assert 0 <= first_cell(Config(1, 4, 0), 13, opening) < 4


def test_eval_defaults_to_random_and_can_reproduce_old_center():
    default = run('exact', 5, BEGINNER, seed_base=202601).results
    random = run('exact', 5, BEGINNER, seed_base=202601, opening='random').results
    assert [(g.won, g.moves) for g in default] == [(g.won, g.moves) for g in random]


def test_cli_reports_opening_and_distinguishes_layout_comparison(tmp_path):
    opts = ['--preset', 'beginner', '--games', '8', '--seed-base', '354712']
    out = tmp_path / 'openings.json'
    assert not main([*opts, '--compare-openings', 'random,center,edge', '--out', str(out)])
    report = json.loads(out.read_text(encoding='utf-8'))
    assert 'paired' not in report
    assert report['openings']['random']['unique_first_cells'] > 1
    assert report['openings']['center']['unique_first_cells'] == 1
    assert 'layouts may differ' in report['comparison']
    assert all(s['games'] == 8 for s in report['openings'].values())
    assert not main([*opts, '--compare', 'exact,endgame', '--out', str(out)])
    report = json.loads(out.read_text(encoding='utf-8'))
    assert report['opening'] == 'random' and report['paired'][0]['pairs'] == 8


def test_cli_rejects_conflicting_opening_options():
    with pytest.raises(SystemExit):
        main(['--opening', 'edge', '--compare-openings', 'random,center'])
