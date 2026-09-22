from hyperopt_inventory import cli


def test_cli_outputs_both_searches(monkeypatch, capsys):
    monkeypatch.setattr(cli, "compare_searches", lambda **_: [
        type("R", (), {"algorithm": "tpe", "reorder_point": 3, "order_up_to": 10, "validation_loss": 1.2})(),
        type("R", (), {"algorithm": "random", "reorder_point": 4, "order_up_to": 11, "validation_loss": 1.4})(),
    ])
    cli.main()
    output = capsys.readouterr().out
    assert "tpe: s=3, S=10" in output
    assert "random: s=4, S=11" in output
