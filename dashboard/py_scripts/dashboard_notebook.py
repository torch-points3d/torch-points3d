#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
import sys
import panel as pn
import numpy as np
import pyvista as pv
pn.extension('vtk')
os.system('/usr/bin/Xvfb :99 -screen 0 1024x768x24 &')
os.environ['DISPLAY'] = ':99'
os.environ['PYVISTA_OFF_SCREEN'] = 'True'
os.environ['PYVISTA_USE_PANEL'] = 'True'


# # Experiment Manager

# In[ ]:


DIR = os.path.dirname(os.getcwd())
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, DIR)


# In[ ]:


from src.visualization.experiment_manager import ExperimentFolder, ExperimentManager


# In[ ]:


experiment_manager = ExperimentManager(DIR)


# In[ ]:


selector_model_name = pn.widgets.Select(name='model_name', options=experiment_manager.model_name_wviz)
selector_paths = pn.widgets.Select(name='model_name_paths')
selector_epoch =  pn.widgets.Select(name='selector_epoch')
selector_split =  pn.widgets.Select(name='selector_split')
selector_file =  pn.widgets.Select(name='selector_file')

camera = [(4.440892098500626e-16, -21.75168228149414, 4.258553981781006),
                       (4.440892098500626e-16, 0.8581731039809382, 0),
                       (0, 0.1850949078798294, 0.982720673084259)]

pl_left = pv.Plotter(notebook=True)
pl_left.camera_position =  camera
pan_left = pn.panel(pl_left.ren_win, sizing_mode='stretch_height', orientation_widget=True)

pl_right = pv.Plotter(notebook=True)
pl_right.camera_position =  camera
pan_right = pn.panel(pl_right.ren_win, sizing_mode='stretch_height', orientation_widget=True)

pl_err = pv.Plotter(notebook=True)
pl_err.camera_position =  camera
pan_err = pn.panel(pl_err.ren_win, sizing_mode='stretch_height', orientation_widget=True)

pans = pn.Row(pan_right, pan_right, pan_err)
col = pn.Column(selector_model_name, selector_paths, selector_epoch, selector_split, selector_file)


# In[ ]:


def update_selector_split(event):
    col[1].options = experiment_manager.get_model_wviz_paths(event)
    col[1].param.trigger('value')
selector_model_name.param.watch(update_selector_split, 'value')
selector_model_name.param.trigger('value')


# In[ ]:


def update_selector_epoch(event):
    col[2].options = experiment_manager.from_paths_to_epoch(event)
    col[2].param.trigger('value')
selector_paths.param.watch(update_selector_epoch, 'value')
selector_paths.param.trigger('value')


# In[ ]:


def update_selector_split(event):
    col[3].options = experiment_manager.from_epoch_to_split(event)
    col[3].param.trigger('value')
selector_epoch.param.watch(update_selector_split, 'value')
selector_epoch.param.trigger('value')


# In[ ]:


def update_selector_file(event):
    col[4].options = experiment_manager.from_split_to_file(event)
    col[4].param.trigger('value')
selector_split.param.watch(update_selector_file, 'value')
selector_split.param.trigger('value')


# In[ ]:


def update_pointcloud(event):
    experiment_manager.load_ply_file(event)
    pl1 = pv.Plotter(notebook=True)
    pl1.camera_position =  [(4.440892098500626e-16, -21.75168228149414, 4.258553981781006),
                           (4.440892098500626e-16, 0.8581731039809382, 0),
                           (0, 0.1850949078798294, 0.982720673084259)]
    point_cloud = pv.PolyData(experiment_manager.current_pointcloud['xyz'])
    point_cloud['l'] = experiment_manager.current_pointcloud['l']
    pl1.add_points(point_cloud)
    pans[0] = pn.panel(pl1.ren_win, sizing_mode='stretch_height', orientation_widget=True)

    pl2 = pv.Plotter(notebook=True)
    pl2.camera_position =  [(4.440892098500626e-16, -21.75168228149414, 4.258553981781006),
                           (4.440892098500626e-16, 0.8581731039809382, 0),
                           (0, 0.1850949078798294, 0.982720673084259)]
    point_cloud = pv.PolyData(experiment_manager.current_pointcloud['xyz'])
    point_cloud['p'] = experiment_manager.current_pointcloud['p']
    pl2.add_points(point_cloud)
    pans[1] = pn.panel(pl2.ren_win, sizing_mode='stretch_height', orientation_widget=True)
    
    pl3 = pv.Plotter(notebook=True)
    pl3.camera_position =  [(4.440892098500626e-16, -21.75168228149414, 4.258553981781006),
                           (4.440892098500626e-16, 0.8581731039809382, 0),
                           (0, 0.1850949078798294, 0.982720673084259)]
    point_cloud = pv.PolyData(experiment_manager.current_pointcloud['xyz'])
    point_cloud['e'] = experiment_manager.current_pointcloud['p'] == experiment_manager.current_pointcloud['l']
    pl3.add_points(point_cloud)
    pans[2] = pn.panel(pl3.ren_win, sizing_mode='stretch_height', orientation_widget=True)
    
    pans[0].jslink(pans[1], camera='camera', bidirectional=True)
    pans[0].jslink(pans[2], camera='camera', bidirectional=True)
    
selector_file.param.watch(update_pointcloud, 'value')
selector_file.param.trigger('value')


# In[ ]:


selector_file.param.trigger('value')
pn.Column(
    '## Select file to display',
    pn.Row(col, pans)
).servable()


# In[ ]:


experiment_manager.current_pointcloud


# In[ ]:


selector_file.param.trigger('value')


# In[ ]:




