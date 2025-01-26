import unittest
from unittest.mock import MagicMock

import numpy as np

from hapsira.render.scene import MainWindow


class TestMainWindow(unittest.TestCase):
    def test_set_model(self):
        main_w = MainWindow()
        main_w.view.add = MagicMock()

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])

        main_w.set_model(vertices, faces)

        # asserting that view.add was called once with proper arguments
        main_w.add.assert_called_once_with(vertices, faces)
