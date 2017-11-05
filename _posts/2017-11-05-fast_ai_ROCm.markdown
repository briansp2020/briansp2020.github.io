---
layout: post
title:  "How to run Fast.ai notebooks using ROCm and TensorFlow."
date:   2017-11-05 21:10:30
categories: ROCm
comments: true
---
Deep learning is all the rage these days. So, I started taking MOOC on deepleaning to learn more about it. [Fast.ai][fast-ai] has a great online class that teaches the subject using a top down approach. It uses Keras and Theano as the main deeplearning toolkit.

One problem with deeplearning field is that it relys too much on one vendor's hardware using a proprietary language. Theano is currently not very well supported using AMD GPU because its developer announced that they won't continue the development. Fortunately, AMD now supports TensorFlow using their ROCm platform. So, I decided to try porting the class notebook to run on ROCm and TensorFlow.

Here are the steps get the class notebook to run on AMD graphics card and ROCm environment. 

### My hardware and software environment

I have Ryzen 7 1800X and Radeon Fury Nano. I tried running [dogs_cats_redux.ipynb][dogs_cats_redux] using CPU only and the finetuning operation took 6 hours using all 8 cores. It was a very painful experience. To make good progress in any field, being able to experiment and iterate fast is a must. So, I decided to try Keras with TensorFlow as the backend since ROCm now support TensorFlow GPU acceration.

### Setting up software environment

Fast.at has AWS instance set up so that anyone who wants to take the class can run the notebook in the cloud. But I prefre running on these softwares on my own machine and I have fast set up at home. No need to pay extra to run it on cloud. Also, porting the class notebook to run on Keras 2.x with TensorFlow backend helped me understand what is going on. You learn more when you are trying to solve problems. :)

In order to use HIP/MIOpen on AMD APUs, you need to install [ROCm][rocm] package. Just follow the instruction and reboot when you are done. Also, Keras and TensorFlow requires some python packages to be installed. So, make sure that you installed the following packages:
{% highlight bash %}
@Kaveri-Ubuntu:~/git$ sudo apt-get install libopencv-dev
{% endhighlight %}

### Porting the class notebook to Keras 2.x and TensorFlow backend

The class notebook was written using Keras 1.x and Theano. Ubuntu 16.04, which ROCm uses has Keras 2.x, which are not 100% backward compatible. Also, models that was trained using Theano backend is not compatible when using TensorFlow and needs to be converted using keras.utils.convert_all_kernels_in_model.

Once you have ROCm and OpenCV set up, you need to clone CS344 repository to your machine:
{% highlight bash %}
@Kaveri-Ubuntu:~/git$ git clone https://github.com/udacity/cs344
{% endhighlight %}

Then change into the directory where the first problem set files are:
{% highlight bash %}
@Kaveri-Ubuntu:~/git$ cd cs344/Problem\ Sets/Problem\ Set\ 1
{% endhighlight %}

HIP has a script file "hipconvertinplace.sh" that will convert all the files in the current directory. The script will print out some information about the files that it converted:
{% highlight bash %}
@Kaveri-Ubuntu:~/git/cs344/Problem Sets/Problem Set 1$ hipconvertinplace.sh 
info: converted 3 CUDA->HIP refs( dev:1 mem:0 kern:1 coord_func:0 math_func:0 special_func:0 stream:0 event:0 err:1 def:0 tex:0 other:0 ) warn:0 LOC:66 in './student_func.cu'
info: converted 8 CUDA->HIP refs( dev:0 mem:8 kern:0 coord_func:0 math_func:0 special_func:0 stream:0 event:0 err:0 def:0 tex:0 other:0 ) warn:0 LOC:84 in './HW1.cpp'
info: converted 2 CUDA->HIP refs( dev:0 mem:0 kern:0 coord_func:0 math_func:0 special_func:0 stream:0 event:0 err:2 def:0 tex:0 other:0 ) warn:0 LOC:88 in './utils.h'
info: converted 4 CUDA->HIP refs( dev:1 mem:2 kern:0 coord_func:0 math_func:0 special_func:0 stream:0 event:0 err:1 def:0 tex:0 other:0 ) warn:0 LOC:93 in './main.cpp'
info: converted 10 CUDA->HIP refs( dev:0 mem:0 kern:0 coord_func:0 math_func:0 special_func:0 stream:0 event:10 err:0 def:0 tex:0 other:0 ) warn:0 LOC:42 in './timer.h'

info: TOTAL-converted 27 CUDA->HIP refs( dev:2 mem:10 kern:1 coord_func:0 math_func:0 special_func:0 stream:0 event:10 err:4 def:0 tex:0 other:0 ) warn:0 LOC:447
  kernels (1 total) :   rgba_to_greyscale(1)
{% endhighlight %}

Once the conversion is done, you still need to tweak a few things.

First, the script will backup the original files with .prehip extension. You do not need those files any more. So, you can delete them:
{% highlight bash %}
@Kaveri-Ubuntu:~/git/cs344/Problem Sets/Problem Set 1$ rm *.prehip
{% endhighlight %}

Second, the converted CUDA file is no longer CUDA file, so its extension should be changed to .cpp:
{% highlight bash %}
@Kaveri-Ubuntu:~/git/cs344/Problem Sets/Problem Set 1$ mv student_func.cu student_func.cpp
{% endhighlight %}

Third, fix up the code to make them compile. There are three things you need to do:
{% highlight bash %}
1. Comment out "#include <cuda.h>" from util.h line 6 and HW1.cpp line 5.
2. Also comment out "checkCudaErrors(hipFree(0));" from HW1.cpp line 27.
3. Add "grid_launch_parm lp" as the first parameter of GPU kernel to student_func.cpp.
{% endhighlight %}

Last, Makefile needs to be adjusted to use HIPCC instead of NVCC:
{% highlight make %}
HIPCC=hipcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

# or if using MacPorts

#OPENCV_LIBPATH=/opt/local/lib
#OPENCV_INCLUDEPATH=/opt/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

HIP_INCLUDEPATH=/opt/rocm/hcc/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

HIPCC_OPTS=-O2 -hc -std=c++amp

student: main.o student_func.o compare.o reference_calc.o Makefile
	$(HIPCC) -o HW1 main.o student_func.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(HIPCC_OPTS)

main.o: main.cpp timer.h utils.h reference_calc.cpp compare.cpp HW1.cpp
	$(HIPCC) -c main.cpp $(HIPCC_OPTS) -I $(HIP_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

student_func.o: student_func.cpp utils.h
	$(HIPCC) -c student_func.cpp $(HIPCC_OPTS)

compare.o: compare.cpp compare.h
	$(HIPCC) -c compare.cpp -I $(OPENCV_INCLUDEPATH) $(HIPCC_OPTS)

reference_calc.o: reference_calc.cpp reference_calc.h
	$(HIPCC) -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(HIPCC_OPTS)

clean:
	rm -f *.o *.png hw
{% endhighlight %}

That's it! Now you should be able to compile and run the program. I have hipified problem set 1 and put them on my Github account [here][hip-ps1]. *Spoiler-Alert* Note that I have added the solution to the problem set 1. Hope this helps.

AMD now has [HIP][amd-hip-github] that allows one to write portable GPGPU code that can be compiled to run on NVidia GPU and AMD GPU. MIOpen library is developing into a good alternative to cuDNN many deeplearning frameworks are in the process of being ported to HIP and MIOpen. You can read more about it [here][amd-deep-learning].

[fast-ai]: http://www.fast.ai
[dogs_cats_redux]: https://github.com/briansp2020/courses/blob/ROCm/deeplearning1/nbs/dogs_cats_redux.ipynb
[amd-hip-github]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP
[amd-deep-learning]: https://rocm.github.io/dl.html
[rocm]: https://github.com/RadeonOpenCompute/ROCm
[opencv]: http://opencv.org/
[opencv-install]: https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-OpenCV-3.1-Installation-Guide
[hip-ps1]: https://github.com/briansp2020/cs344/tree/master/Problem%20Sets/Problem%20Set%201
