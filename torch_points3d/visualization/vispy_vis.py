import vispy
from vispy.scene import visuals, SceneCanvas

class Vispy:
    def __init__(self, use_default=True, mode="sem_kitti"):
        if use_default:
            self.setup_vispy(mode)
        else:
            print("Using custom vispy layout!")

    def setup_vispy(self, mode):
        if mode == "sem_kitti":
            self.canvas = SceneCanvas(keys='interactive', show=True)
            self.grid = self.canvas.central_widget.add_grid()
            
            self.pc_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
            self.grid.add_widget(self.pc_view, 0, 0, row_span=5)
            self.pc_vis = visuals.Markers()
            self.pc_view.camera = 'turntable'
            self.pc_view.add(self.pc_vis)
            visuals.XYZAxis(parent=self.pc_view.scene)

            self.proj_img_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
            self.grid.add_widget(self.proj_img_view, 5, 0)
            self.img_vis = visuals.Image()
            self.proj_img_view.add(self.img_vis)

    def update(self, event):
        pass
    
    def run(self):
        timer = vispy.app.Timer()
        timer.connect(self.update)
        timer.start(0.1)
        vispy.app.run()