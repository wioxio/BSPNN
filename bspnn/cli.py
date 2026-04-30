"""
CLI wrappers for executing BSPNN step modules.
"""

import runpy


def step1():
    runpy.run_module("bspnn.steps.step1_primary_prediction", run_name="__main__")


def step2():
    runpy.run_module("bspnn.steps.step2_prediction_level1", run_name="__main__")


def step3():
    runpy.run_module("bspnn.steps.step3_prediction_level2", run_name="__main__")
