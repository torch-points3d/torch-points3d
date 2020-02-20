# S3DIS

For S3DIS we use the original dataset for which each area can be fused back together into larger zones. It corresponds to the `Stanford3dDataset_v1.2.zip` file when you are in the download folder.

The dataset contains some bugs which can be found and corrected at those position.
We provide a patch to fix those. First copy the file s3dis.patch in your s3dis root folder that should contain a `raw` subfolder. Mine looks like that:

```bash
drwxrwxr-x 5 nicolas nicolas       4096 Feb 20 21:40 ./
drwxrwxr-x 4 nicolas nicolas       4096 Feb 20 19:49 ../
-rw-rw-r-- 1 nicolas nicolas 5149528258 Feb 20 18:12 Stanford3dDataset_v1.2.zip
drwxrwxr-x 2 nicolas nicolas       4096 Feb 20 20:16 processed/
drwxrwxr-x 8 nicolas nicolas       4096 Feb 20 18:44 raw/
-rw-rw-r-- 1 nicolas nicolas       1453 Feb 20 21:41 s3dis.patch
```

Then run the following command:

```bash
patch -ruN -p0 -d  raw < s3dis.patch
```

It should return something like that:

```bash
patching file Area_1/WC_1/WC_1.txt
patching file Area_6/hallway_2/hallway_2.txt
```
