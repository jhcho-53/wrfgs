from argparse import ArgumentParser
from arguments import ModelParams


def test_datadir_arg_threads_through_extract():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args(["--datadir", "/tmp/converted_scene"])
    g = lp.extract(args)
    assert g.datadir == "/tmp/converted_scene"


def test_datadir_has_default():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args([])
    g = lp.extract(args)
    assert g.datadir == "./data_test200"
