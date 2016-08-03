# ss-gan

This code is developed based on eyescream project with Torch: [project site](https://github.com/facebook/eyescream).

This code is the implementation of training and testing for S^2-GAN for the following paper:

**Xiaolong Wang** and Abhinav Gupta. Generative Image Modeling using Style and Structure Adversarial Networks. Proc. of European Conference on Computer Vision (ECCV), 2016. [pdf](http://arxiv.org/pdf/1603.05631.pdf)

BibTeX: 
```txt
@inproceedings{Wang_SSGAN2016,
    Author = {Xiaolong Wang and Abhinav Gupta},
    Title = {Generative Image Modeling using Style and Structure Adversarial Networks},
    Booktitle = {ECCV},
    Year = {2016},
}
```



Models and Datasets
----

The trained models can be downloaded from [dropbox](https://www.dropbox.com/sh/zz7v8gfmgjvswxx/AACdZC045j88zHRnGyBIxuj_a?dl=0). 

The pre-processed dataset (NYUv2) including rgb images and TV-denoised Surface Normals in jpgs can be downloaded from [dropbox](). 

The list of training files [dropbox](https://dl.dropboxusercontent.com/u/334666754/ssgan/trainlist_rand.txt). 

General Instructions for using the code
----


For training, one need to:
```txt
	Update the path_dataset = '/scratch/xiaolonw/render_data/' in dataset.lua 
	Update the opt.save in train.lua for saving models 
```

For testing, one can download the models into the ssgan_models folder. 

Structure-GAN 
----

The code for Stucture-GAN is in structure-gan:
```txt
	train.lua: training Stucture-GAN
	test.lua: testing Stucture-GAN
	ssgan_models/Structure_GAN.net is our trained model
```

Style-GAN 
----

The code for Style-GAN without FCN constraints is in style-gan-nofcn:
```txt
	train.lua: training Style-GAN
	test.lua: testing Style-GAN (To run this you need to download the dataset)
	ssgan_models/Style_GAN_nofcn.net is our trained model
```

The code for Style-GAN with FCN constraints is in style-gan-fcn:
```txt
	train_fcn.lua: training FCN for surface normal estimation
	test_fcn.lua: testing FCN for surface normal estimation (To run this you need to download the dataset)
	ssgan_models/FCN.net is our trained model

	train_gan.lua: training Style-GAN
	test_gan.lua: testing Style-GAN (To run this you need to download the dataset)
	ssgan_models/joint_Style_GAN.net is our trained model
```

Joint Learning for S^2-GAN
----

The code for joint learning is in joint-ssgan:
```txt
	train.lua: joint learning 
	test.lua: testing S^2-GAN
```













