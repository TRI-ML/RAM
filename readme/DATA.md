# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation or training, you will need to setup datasets.

### ParallelDomain (PD)

This is the synthetic dataset proposed in [PermaTrack](https://github.com/TRI-ML/permatrack). You can download the images together with annotations under this [link](https://tri-ml-public.s3.amazonaws.com/datasets/pd_release.tar.gz). After downloading, copy the contents into `$RAM_ROOT/data/pd`.

### KITTI Tracking

We use KITTI Tracking to train and evaluate the system in the real world. Following prior work, we will only use the training set (and create a validation set from it) for developing this project.

- Download [images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip), and [annotations](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) from [KITTI Tracking website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and unzip. Place or symlink the data as below:

  ~~~
  ${PermaTrack_ROOT}
  |-- data
  `-- |-- kitti_tracking
      `-- |-- data_tracking_image_2
          |   |-- training
          |   |-- |-- image_02
          |   |-- |-- |-- 0000
          |   |-- |-- |-- ...
          |-- |-- testing
          |-- label_02
          |   |-- 0000.txt
          |   |-- ...
  ~~~

- Run `python convert_kittitrack_to_coco.py` in `tools` to convert the annotation into COCO format. 
- The resulting data structure should look like:

  ~~~
  ${PermaTrack_ROOT}
  |-- data
  `-- |-- kitti_tracking
      `-- |-- data_tracking_image_2
          |   |-- training
          |   |   |-- image_02
          |   |   |   |-- 0000
          |   |   |   |-- ...
          |-- |-- testing
          |-- label_02
          |   |-- 0000.txt
          |   |-- ...
          |-- data_tracking_calib
          |-- label_02_val_half
          |   |-- 0000.txt
          |   |-- ...
          |-- label_02_train_half
          |   |-- 0000.txt
          |   |-- ...
          `-- annotations
              |-- tracking_train.json
              |-- tracking_test.json
              |-- tracking_train_half.json
              `-- tracking_val_half.json
  ~~~

To convert the annotation in a suitable format for evaluating track AP, run this command in `tools`: `python convert_kitti_to_tao.py` 

### LA-CATER

Is a toy synthetic obejct permanence benchmark. You can download the frames from the corresponding [project web page](https://chechiklab.biu.ac.il/~avivshamsian/OP/OP_HTML.html). After downloading, copy the contents into `$RAM_ROOT/data/la_cater`. The annotations are avaiable under this [link]() and should be palced under `$RAM_ROOT/data/la_cater/annotations`.

### LA-CATER-Moving

Is our extrension of LA-CATER with a moving camera. You can download the frames and annotations under this [link](). After downloading, copy the contents into `$RAM_ROOT/data/la_cater_moving`. 


## References
Please cite the corresponding References if you use the datasets.

~~~
@inproceedings{tokmakov2021learning,
  title={Learning to Track with Object Permanence},
  author={Tokmakov, Pavel and Li, Jie and Burgard, Wolfram and Gaidon, Adrien},
  booktitle={ICCV},
  year={2021}
}

@INPROCEEDINGS{Geiger2012CVPR,
    author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {CVPR},
    year = {2012}
}

@inproceedings{shamsian2020learning,
  title={Learning Object Permanence from Video},
  author={Shamsian, Aviv and Kleinfeld, Ofri and Globerson, Amir and Chechik, Gal},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{girdhar2019cater,
  title={{CATER}: A diagnostic dataset for Compositional Actions and TEmporal Reasoning},
  author={Girdhar, Rohit and Ramanan, Deva},
  booktitle={ICLR},
  year={2020}
}
~~~