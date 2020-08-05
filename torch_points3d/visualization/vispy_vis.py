import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np

class Vispy:
    def __init__(self, mode="sem_kitti", dataset=None, model=None):
        self.dataset = dataset
        self.model = model
        self.block = False
        self.counter = 0
        self.mode = mode
        if mode=="custom":
            print("vispy canvas needs to be setup manually")
        else:
            self.setup_vispy(mode)
    
    def setup_vispy(self, mode):
        if mode == "sem_kitti":
            self.setup_semkitti()
    
    def setup_semkitti(self):
        self.canvas = SceneCanvas(keys="interactive", show=True)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)

        self.grid = self.canvas.central_widget.add_grid()
        self.pc_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.pc_view, 0, 0, row_span=5)
        self.pc_vis = visuals.Markers()
        self.pc_view.camera = 'turntable'
        self.pc_view.add(self.pc_vis)
        visuals.XYZAxis(parent=self.pc_view.scene)

        if self.model is not None or self.dataset.label_path is not None:
            self.proj_img_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene, size=(self.sensor["img_prop"]["width"], self.sensor["img_prop"]["height"]))
            self.grid.add_widget(self.proj_img_view, 5, 0)
            self.img_vis = visuals.Image()
            self.proj_img_view.add(self.img_vis)
    
    def update_semkitti(self):
        pts, sem_label, proj, proj_label, proj_x, proj_y = self.dataset.project_returnall(self.counter)
        pt_color = np.ones((pts.shape[0], 3), dtype=np.float)*80/255 # gray color
        img_color = None
        if self.model is not None:
            pred_labels = self.model(proj).argmax(dim=1).cpu().numpy()[0]
            img_color = self.dataset.colorise(pred_labels)
            unproj_pred = pred_labels[proj_y, proj_x]
            pt_color = self.dataset.colorise(unproj_pred)
        elif self.sem_label is not None:
            pt_color = self.dataset.colorise(sem_label)
            img_color = self.dataset.colorise(proj_label)
        
        self.pc_vis.set_data(pts, face_color=pt_color, edge_color=pt_color, size=2)
        if img_color is not None:
            self.img_vis.set_data(img_color)

    def update(self, event=None):
        if self.mode == "sem_kitti":
            self.update_semkitti()
        if event is not None:
            self.counter +=1       
    
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
        elif event.key == 'Escape':
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