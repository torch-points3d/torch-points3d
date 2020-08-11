import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.models.base_model import BaseModel


class BaseScanAndImgVisualizer(object):
    def __init__(self, width=None, height=None):
        self.block = False
        self.counter = 0
        self.width = width
        self.height = height
        self.setup_vispy()

    def setup_vispy(self):
        self.canvas = SceneCanvas(keys="interactive", show=True)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)

        self.grid = self.canvas.central_widget.add_grid()
        self.pc_view = vispy.scene.widgets.ViewBox(border_color="white", parent=self.canvas.scene)
        self.grid.add_widget(self.pc_view, 0, 0, row_span=5)
        self.pc_vis = visuals.Markers()
        self.pc_view.camera = "turntable"
        self.pc_view.add(self.pc_vis)
        visuals.XYZAxis(parent=self.pc_view.scene)

        if self.height is not None and self.width is not None:
            self.proj_img_view = vispy.scene.widgets.ViewBox(
                border_color="white", parent=self.canvas.scene, size=(self.width, self.height)
            )
            self.grid.add_widget(self.proj_img_view, 5, 0)
            self.img_vis = visuals.Image()
            self.proj_img_view.add(self.img_vis)

    def update_app(self):
        raise NotImplementedError

    def update(self, event=None):
        self.update_app()
        if event is not None:
            self.counter += 1

    def key_press(self, event):
        if self.block:
            if event.key == "Space":
                self.block = False
                self.timer.start(0.1)
            elif event.key == "Up":
                self.counter += 1
                self.update()
            elif event.key == "Down":
                self.counter -= 1
                self.update()
        elif self.block == False:
            if event.key == "Space":
                self.block = True
                self.timer.stop()
        elif event.key == "Escape":
            self.destroy()

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def destroy(self):
        self.canvas.close()
        vispy.app.quit()

    def run(self):
        self.timer = vispy.app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0.1)
        vispy.app.run()


class KittiVisualizer(BaseScanAndImgVisualizer):

    MODES = ["errors", "preds", "labels"]

    def __init__(self, dataset: BaseDataset = None, model: BaseModel = None, sensor=None, mode="errors"):
        assert mode in MODES
        assert dataset is not None
        if mode in ["errors", "preds"]:
            assert model is not None
        width = None
        height = None
        if sensor is not None:
            width = sensor["img_prop"]["width"]
            height = sensor["img_prop"]["height"]
        BaseScanAndImgVisualizer.__init__(width=width, height=height)
        self.dataset = dataset
        self.model = model
        self.mode = mode

    def _update_app(self):
        data = self.dataset[self.counter]
        pt_color = np.ones((data.pos.shape[0], 3), dtype=np.float) * (80 / 255.0)  # gray color
        img_color = None
        if self.mode == "preds":
            pred_labels = self.model(data.proj).argmax(dim=1).cpu().numpy()[0]
            img_color = self.dataset.colorise(pred_labels)
            unproj_pred = pred_labels[data.proj_y, data.proj_x]
            pt_color = self.dataset.colorise(unproj_pred)
        elif self.mode == "labels":
            pt_color = self.dataset.colorise(data.sem_label)
            img_color = self.dataset.colorise(data.proj_label)
        elif self.mode == "errors":
            raise NotImplementedError

        self.pc_vis.set_data(data.pos, face_color=pt_color, edge_color=pt_color, size=2)
        if img_color is not None:
            self.img_vis.set_data(img_color)
            self.canvas.events.key_press.unblock()
