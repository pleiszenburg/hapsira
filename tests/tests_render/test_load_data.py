from io import StringIO

import numpy as np

from hapsira.render.data_loader import load_data


def test_load_data():
    data = """v 1.0 2.0 3.0
              v 4.0 2.0 3.0
              f 2   2   4"""

    file_path = StringIO(data)

    test_vertices = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 3.0]])

    # zero-index face
    test_faces = np.array([[1, 1, 3]])

    vertices, faces = load_data(file_path)

    np.testing.assert_array_equal(vertices, test_vertices)
    np.testing.assert_array_equal(faces, test_faces)


test_load_data()
