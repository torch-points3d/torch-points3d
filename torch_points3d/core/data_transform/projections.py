import numpy as np

class SphericalProjection:
    """
    This class projects points with its labels into a 360 img.
    in velodyne coordinates - forward=x, left=y, up=z
    in image coordinates -  ----> x-axis
                            |
                            |
                            v y-axis
    """
    def __init__(self, img_H, img_W, fov_UP=-3, fov_DOWN=25, normalise=True, means=None, std=None):
        self._img_H = img_H
        self._img_W = img_W
        self._fov_UP = fov_UP/180*np.pi
        self._fov_down = fov_DOWN/180*np.pi
        self._fov = abs(self.fov_UP)+abs(self.fov_DOWN)
        self._normalise = normalise
        self._means = np.array(means, dtype=np.float)
        self._std = np.array(std, dtype=np.float)
    
    def _project_scan(self, scan):
        pos = scan[:, 0:3]
        remissions = scan[:, 3]
        # calculate depth
        depth = np.linalg.norm(pos, 2, axis=1)
        scan_x, scan_y, scan_z = pos[:, 0], pos[:, 1], pos[:, 2]

        # yaw angle -> -tan-1(left/forward) -> (-pi,pi)
        # pitch angle -> sin-1(top/depth) -> (-fov_down, fov_up)
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z/depth)

        """
        mapping indices to image
        convert yaw and pitch to width and height ranges 
        convert to closest integer and ensure end limits are not crossed
        """
        proj_x = 0.5*(yaw/np.pi+1.0)
        proj_x = np.floor(proj_x*self._img_W)
        proj_x = np.minimum(self._img_W-1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32) 

        proj_y = (abs(self._fov_UP)-pitch)/self._fov
        proj_y = np.floor(proj_y*self._img_H)
        proj_y = np.minimum(self._img_H-1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32) 
        
        # order by decreasing depth (ensures closer pos are preferred over farther ones)
        indices = np.arange(depth.shape[0]) 
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pos = pos[order]
        remissions = remissions[order]
        proj_x = proj_x[order]
        proj_y = proj_y[order]

        # initializing empty arrays
        proj_range = np.full((self._img_H, self._img_W), -1, dtype=np.float32)
        proj_xyz = np.full((self._img_H, self._img_W, 3), -1, dtype=np.float32)
        proj_remissions = np.full((self._img_H, self._img_W), -1, dtype=np.float32)
        proj_index = np.full((self._img_H, self._img_W), -1, dtype=np.int32)

        # assigning values!
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = pos
        proj_remissions[proj_y, proj_x] = remissions
        proj_index[proj_y, proj_x] = indices
        proj_mask = (proj_index >= 0).astype(np.bool)

        return pos, remissions, proj_range, proj_xyz, proj_remissions, proj_x, proj_y, order, proj_index, proj_mask
    
    def _project_labels(self, proj_index, proj_mask, unproj_labels):
        proj_labels = np.zeros((self.img_H, self.img_W), dtype=np.int32)
        proj_labels[proj_mask] = unproj_labels[proj_index[proj_mask]]
        return proj_labels
    
    def __call__(self, data):
        data.pos, data.remissions, proj_range, proj_xyz, proj_remissions, data.proj_x, data.proj_y, data.order, \
            data.proj_index, data.proj_mask = self._project_scan(data.scan)
        data.proj = np.concatenate([np.expand_dims(proj_range, 0), np.transpose(proj_xyz, (2, 0, 1)), np.expand_dims(proj_remissions, 0)])
        if self._normalise:
            data.proj = (data.proj - self.means[:, None, None])/self.std[:, None, None]
        data.proj_labels = None
        if data.labels is not None:
            data.proj_labels = self._project_labels(data.proj_index, data.proj_mask, data.labels)
        return data
        

        