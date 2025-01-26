from vispy import scene
from vispy.geometry import MeshData
from vispy.scene import visuals


class MainWindow(scene.SceneCanvas):
    """
    Main window for rendering 3D models using Vispy.

    This class provides a graphical interface for rendering 3D models using the Vispy library.
    It sets up a scene canvas with interactive capabilities, allowing users to visualize and
    interact with 3D models. The canvas includes a grid layout and a view with a turntable
    camera for easy manipulation of the 3D scene.

    Parameters
    ----------
    *args : tuple
        Additional arguments for the SceneCanvas.
    """

    def __init__(self, *args):
        super().__init__(keys="interactive", size=(800, 600), show=True)
        self.unfreeze()
        self.grid = self.central_widget.add_grid(margin=10)
        self.view = self.grid.add_view(row=0, col=0, camera="turntable")
        self.view.bgcolor = "black"
        self.view.camera.fov = 60

    def set_model(self, vertices, faces):
        """
        Set the 3D model to be rendered.

        This method takes arrays of vertices and faces, creates a mesh from them, and adds
        the mesh to the view for rendering. The mesh is shaded smoothly and colored grey.

        Parameters
        ----------
        vertices : numpy.ndarray
            Array of vertex coordinates.
        faces : numpy.ndarray
            Array of face indices.
        """
        mesh_data = MeshData(vertices=vertices, faces=faces)
        mesh = visuals.Mesh(meshdata=mesh_data, shading="smooth", color="grey")
        self.view.add(mesh)
