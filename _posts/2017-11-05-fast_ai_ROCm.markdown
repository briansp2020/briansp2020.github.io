---
layout: post
title:  "How to run Fast.ai notebooks using ROCm and TensorFlow."
date:   2017-11-05 21:10:30
categories: ROCm
comments: true
---
Deep learning is all the rage these days. So, I started taking MOOC on deep learning to learn more about it. [Fast.ai][fast-ai] has a great online class that teaches the subject using a top down approach. Its first class uses Keras and Theano as the main deep learning toolkit.

One problem in deep learning field has been that it relied too much on one vendor's hardware using a proprietary programming language and library. Though the toolkits are open source and are freely available, in order to run them efficiently, you needed NVidia GPU with CUDA and cuDNN, neither of them are open or cross platform across different hardware platform. Not any more. Now you have choices.

I'm a big fan of AMD and their open approach to GPU programming. So, I decided to port the class notebook to run on AMD ROCm platform. There were two main issues that I needed to resolve. First, Keras on my machine running Ubuntu 16.04 was version 2.x. There were minor API changes between 1.x and 2.x. Error messages from Keras library was very helpful in resolving the incompatibilities. The second issue was changing the backend from Theano to TensorFlow. Theano is no longer in active development and is not available on ROCm platform. Fortunately, AMD now supports TensorFlow on their ROCm platform. However, running a model trained using Theano does not work well if the backend is changed to TensorFlow. The model needs to be converted using keras.utils.convert_all_kernels_in_model.

Here are the steps to get the class notebooks to run on AMD graphics card and ROCm environment. 

### My hardware and software environment

I have Ryzen 7 1800X and Radeon Fury Nano. I tried running [dogs_cats_redux.ipynb][dogs_cats_redux] using CPU only and the fine-tuning operation took 6 hours using all 8 cores. It was a very painful experience. To make good progress in any field, being able to experiment and iterate fast is a must.

Fast.at has AWS instance set up so that anyone who wants to take the class can run the notebook in the cloud. But I prefer running these softwares on my own machine. I have a fast set up at home. No need to pay extra to run it on cloud. Also, porting the class notebook to run on Keras 2.x with TensorFlow backend helped me understand what is going on better. You learn more when you are trying to solve problems. :)

In order to use HIP/MIOpen on AMD hardware, you need to install [ROCm][rocm] package. Just follow the instruction and reboot when you are done.

Also, Keras and TensorFlow requires some python packages to be installed. So, make sure that you install the following packages:
{% highlight bash %}
@Ryzen1800X:~$ sudo apt-get install -y \
    git \
    python-numpy \
    python-dev \
    python-wheel \
    python-mock \
    python-future \
    python-pip \
    python-yaml \
    python-setuptools \
    python-matplotlib \
    python-pandas \
    python-sklearn \
    python-h5py \
    ipython \
    ipython-notebook
{% endhighlight %}

To install jupyter you must upgrade pip. While you are at it, install bcolz, and keras using pip as well. Installation of bcolz will take a while. So, be patient.
{% highlight bash %}
@Ryzen1800X:~$ sudo -H pip install --upgrade pip
@Ryzen1800X:~$ sudo -H pip install jupyter bcolz keras
{% endhighlight %}

TensorFlow uses [Eigen][eigen]. Install [hipEigen][hipeigen] using the following command:
{% highlight bash %}
@Ryzen1800X:~$ sudo git clone -b develop https://github.com/ROCmSoftwarePlatform/hipeigen.git /opt/rocm/hipeigen
{% endhighlight %}

Hip version of [TensorFlow][amd-tensorflow] package is available on repo.radeon.com. Install it using the following commands:
{% highlight bash %}
@Ryzen1800X:~$ wget http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl
@Ryzen1800X:~$ sudo -H pip install tensorflow-1.0.1-cp27-cp27mu-linux_x86_64.whl
{% endhighlight %}

You also need MIOpen and BLAS libraries.
{% highlight bash %}
@Ryzen1800X:~$ sudo apt-get install -y miopengemm miopen-hip hipblas rocblas
{% endhighlight %}

More detailed instructions are available on [AMD's GitHub page][amd-tensorflow].

### Porting the class notebook to Keras 2.x and TensorFlow backend

The class notebook was written using Keras 1.x and Theano. Ubuntu 16.04, which ROCm uses has Keras 2.x, which are not 100% backward compatible. Also, models that was trained using Theano backend is not compatible when using TensorFlow and needs to be converted using keras.utils.convert_all_kernels_in_model. I already did the conversion and put them on my [GitHub account][converted-notebook]. Make sure to try ROCm branch.

The fine-tuning 1 epoch in [dogs_cats_redux.ipynb][dogs_cats_redux] now takes about 3.5 minutes using Fury Nano GPU instead of 2 hours using Ryzen 7 1800X CPU. Running all 3 epoch takes a little over 10 minutes instead of 6 hours! Boy, am I glad that my GPU works for deep learning training!

### To the infinity and beyond

AMD now has [HIP][amd-hip-github] that allows one to write portable GPGPU code that can be compiled to run on NVidia GPU and AMD GPU. [MIOpen][miopen] library is developing into a good alternative to cuDNN and many deep learning frameworks are in the process of being ported to HIP and MIOpen. You can read more about it [here][amd-deep-learning].

I plan on converting and testing all the class notebooks on my machine. I'll update my GitHub repo as I test and verify that the notebooks work on my machine.

[fast-ai]: http://www.fast.ai
[dogs_cats_redux]: https://github.com/briansp2020/courses/blob/ROCm/deeplearning1/nbs/dogs_cats_redux.ipynb
[amd-hip-github]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP
[amd-deep-learning]: https://rocm.github.io/dl.html
[eigen]: http://eigen.tuxfamily.org/index.php?title=Main_Page
[converted-notebook]: https://github.com/briansp2020/courses
[amd-tensorflow]: https://github.com/ROCmSoftwarePlatform/hiptensorflow/blob/hip/README.ROCm.md
[rocm]: https://github.com/RadeonOpenCompute/ROCm
[miopen]: https://github.com/ROCmSoftwarePlatform/MIOpen
[hipeigen]: https://github.com/ROCmSoftwarePlatform/hipeigen
