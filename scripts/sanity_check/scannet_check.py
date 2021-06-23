import os
import glob
import argparse


def main(config):
    dataset_base_dir = config.base_dir
    scene_list = os.listdir(dataset_base_dir)
    for scene in scene_list:
        scene_raw = scene
        scene = os.path.join(dataset_base_dir, scene, scene_raw)
        check_aggregation_json = os.path.isfile(scene + ".aggregation.json")
        check_gt = os.path.isfile(scene + ".txt")
        check_seg = os.path.isfile(scene + "_vh_clean_2.0.010000.segs.json")
        check_ply = os.path.isfile(scene + "_vh_clean_2.ply")
        if check_aggregation_json and check_gt and check_seg and check_ply:
            # print("success scene: {}".format(scene_raw))
            pass
        else:
            print("error scene: {} | agg: {}, gt: {}, seg: {}, ply: {}".format(scene_raw, check_aggregation_json,
                                                                               check_gt, check_seg, check_ply))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sanity check for ScanNet.')
    parser.add_argument('--base_dir', type=str, default="../data/scannet-sparse/raw/scans/",
                        help='dataset sequence base directory')
    args = parser.parse_args()
    main(args)
