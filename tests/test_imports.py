import pytest

from physics import rk4, neural
import models.train_neural_generator


def test_physics_modules_exist():
    assert hasattr(rk4, 'run_kepler_simulation')
    assert hasattr(rk4, 'rk4_step')
    assert hasattr(neural, 'NeuralIntegrator')


def test_training_script_import():
    # simply ensure the module loads without error
    assert models.train_neural_generator is not None
