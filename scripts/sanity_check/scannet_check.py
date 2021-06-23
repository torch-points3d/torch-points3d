import os
import glob
import argparse
import tempfile
import urllib
from urllib.request import urlopen

def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print("\t" + url + " > " + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, "w")
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        # urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        pass
        # log.warning("WARNING Skipping download of existing file " + out_file)

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
            print("re-downloading the missing file...")
            if config.version == "v2":
                base_url = "http://kaldir.vc.in.tum.de/scannet/v2/scans"
            elif config.version == "v1":
                base_url = "http://kaldir.vc.in.tum.de/scannet/v1/scans"
            else:
                print("error, not supported version: {}".format(config.version))
                break
            if not check_aggregation_json:
                download_link = base_url + "/" + scene_raw + "/" + scene_raw + ".aggregation.json"
                out_filename = scene + ".aggregation.json"
                download_file(download_link, out_filename)
            if not check_gt:
                download_link = base_url + "/" + scene_raw + "/" + scene_raw + ".txt"
                out_filename = scene + ".txt"
                download_file(download_link, out_filename)
            if not check_seg:
                download_link = base_url + "/" + scene_raw + "/" + scene_raw + "_vh_clean_2.0.010000.segs.json"
                out_filename = scene + "_vh_clean_2.0.010000.segs.json"
                download_file(download_link, out_filename)
            if not check_ply:
                download_link = base_url + "/" + scene_raw + "/" + scene_raw + "_vh_clean_2.ply"
                out_filename = scene + "_vh_clean_2.ply"
                download_file(download_link, out_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sanity check for ScanNet.')
    parser.add_argument('--base_dir', type=str, default="../data/scannet-sparse/raw/scans/",
                        help='dataset sequence base directory')
    parser.add_argument("--version", type=str, default="v2", help="version of scanNet, (default is v2)")
    args = parser.parse_args()
    main(args)
