from torch_geometric.data import InMemoryDataset
from torch_points3d.datasets.segmentation.scannet import *
from torch_points3d.metrics.object_detection_tracker import ObjectDetectionTracker

class ScannetObjectDetection(Scannet):

    MAX_NUM_OBJ = 64
    NUM_CLASS = 18
    NUM_HEADING_BIN = 1
    NUM_SIZE_CLUSTER = 18
    TYPE2CLASS = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
        'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
        'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17}  
    NYU40IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
    MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

    def __init__(self, *args, **kwargs):
        super(ScannetObjectDetection, self).__init__(*args, **kwargs)

        self.CLASS2TYPE = {self.TYPE2CLASS[t]:t for t in self.TYPE2CLASS}
        self.NYU40ID2CLASS = {nyu40id: i for i,nyu40id in enumerate(list(self.NYU40IDS))}

    def get(self, idx):
        data = super(ScannetObjectDetection, self).get(idx)

        target_bboxes = torch.zeros((self.MAX_NUM_OBJ, 6))
        target_bboxes_mask = torch.zeros((self.MAX_NUM_OBJ))    
        angle_classes = torch.zeros((self.MAX_NUM_OBJ,))
        angle_residuals = torch.zeros((self.MAX_NUM_OBJ,))
        size_classes = torch.zeros((self.MAX_NUM_OBJ,))
        size_residuals = torch.zeros((self.MAX_NUM_OBJ, 3))

        num_points = data.pos.shape[0]
        semantic_labels = data.y
        instance_labels = data.instance_labels
        instance_bboxes = data.instance_bboxes

        bbox_mask = np.in1d(instance_bboxes[:,-1], self.NYU40IDS)
        instance_bboxes = instance_bboxes[bbox_mask,:]

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_votes = torch.zeros([num_points, 3])
        point_votes_mask = torch.zeros(num_points)
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label 
            if semantic_labels[ind[0]].item() in self.NYU40IDS:
                x = data.pos[ind,:3]
                center = 0.5*(x.min(0)[0] + x.max(0)[0])
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = point_votes.repeat((1, 3)) # make 3 votes identical 

        class_ind = np.asarray([np.where(self.NYU40IDS == x)[0][0] for x in np.asarray(instance_bboxes[:,-1])])

        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = torch.from_numpy(class_ind)
        #size_residuals[0:instance_bboxes.shape[0], :] = \
        #    target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]
            

        target_bboxes_semcls = np.zeros((self.MAX_NUM_OBJ))                                
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [self.NYU40ID2CLASS[x.item()] for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]  

        data.center_label = target_bboxes.float()[:,0:3]
        data.heading_class_label = angle_classes.int()
        data.heading_residual_label = angle_residuals.float()
        data.heading_class_label = angle_classes.int()
        data.size_class_label = size_classes.int()
        data.size_residual_label = size_residuals.float()
        data.sem_cls_label = torch.from_numpy(target_bboxes_semcls).int()
        data.box_label_mask = target_bboxes_mask.float()
        data.vote_label = point_votes.float()
        data.vote_label_mask = point_votes_mask.int()
        data.vote_label = point_votes.float()
        delattr(data, "instance_bboxes")

        return data


class ScannetDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        use_instance_labels: bool = dataset_opt.use_instance_labels
        use_instance_bboxes: bool = dataset_opt.use_instance_bboxes
        donotcare_class_ids: [] = dataset_opt.donotcare_class_ids if dataset_opt.donotcare_class_ids else []
        max_num_point: int = dataset_opt.max_num_point if dataset_opt.max_num_point != "None" else None

        self.train_dataset = ScannetObjectDetection(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
        )

        self.val_dataset = ScannetObjectDetection(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
        )

    @staticmethod
    def get_tracker(model, dataset, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return ObjectDetectionTracker(
            dataset, wandb_log=wandb_log, use_tensorboard=tensorboard_log
        )