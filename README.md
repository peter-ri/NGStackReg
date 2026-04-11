# NGStackReg

NGStackReg is a parallelized, OpenCL accelerated rewrite of the 
[StackReg](http://bigwww.epfl.ch/thevenaz/stackreg/) and [TurboReg](http://bigwww.epfl.ch/thevenaz/turboreg/) 
plugins that enables the registration of multidimensional image stacks. It was written with memory efficiency 
and registration speed in mind. Depending on the hardware quite significant speedups of >20x can be achieved 
compared to the original plugins. For images with \<=16-bit depth, the GPU single precision mode is recommended; 
otherwise, the GPU hybrid precision mode is recommended.

## Description

This plugin is meant to alleviate some inconveniences of the otherwise excellent 
[StackReg](http://bigwww.epfl.ch/thevenaz/stackreg/) and [TurboReg](http://bigwww.epfl.ch/thevenaz/turboreg/) 
plugins [(Th&eacute;venaz et al., 1998)](http://dx.doi.org/10.1109/83.650848). The inability of these plugins 
to align along a chosen axis poses a problem for typical multidimensional stacks, because each dimension needs 
to be aligned separately. Since [StackReg](http://bigwww.epfl.ch/thevenaz/stackreg/) passes each frame to 
[TurboReg](http://bigwww.epfl.ch/thevenaz/turboreg/) via a temporary image file with a hardcoded name, 
alignments cannot be run in parallel, as already pointed out by the authors. In addition to the computational 
burden, aligning each axis separately may lead to disparities between the alignments. 
Some of these limitations have been addressed by earlier projects, including:

* PoorMan3Dreg (Liebling, 2010), 

* [HyperStackReg (Sharma, 2015)](https://github.com/ved-sharma/HyperStackReg), 

* [MultiStackRegistration (Busse and Miura, 2016)](https://github.com/miura/MultiStackRegistration) and 

* [TimeLapseReg (Sahdev et al., 2017)](https://github.com/incfbelgiannode/TimeLapseReg). 

However, these projects still rely on the proven [TurboReg](http://bigwww.epfl.ch/thevenaz/turboreg/) base 
[(Th&eacute;venaz et al., 1998)](http://dx.doi.org/10.1109/83.650848).


NGStackReg is a complete rewrite of the aforementioned [StackReg](http://bigwww.epfl.ch/thevenaz/stackreg/) and 
[TurboReg](http://bigwww.epfl.ch/thevenaz/turboreg/) plugins and is based on the papers written by 
[Th&eacute;venaz et al. (1998)](http://dx.doi.org/10.1109/83.650848), 
Unser et al. ([1993a](http://dx.doi.org/10.1109/78.193220), [1993b](http://dx.doi.org/10.1109/78.193221), 
[1999](http://dx.doi.org/10.1109/79.799930)) and 
[Ruijters et al. (2012)](https://doi.org/10.1093/comjnl/bxq086). It was partially written 
in the laboratory of [Prof. Basler](https://www.biozentrum.unibas.ch/research/research-groups/research-groups-a-z/overview/unit/research-group-marek-basler) 
at the Biozentrum of the University of Basel [(Ringel, 2018)](https://doi.org/10.5451/unibas-006805400).

The plugin allows selection of the channel, Z or time axis as the alignment axis. The currently selected 
stack position is used as the alignment reference. In case an image in the center of 
the stack is of particular interest, just select it and everything will be aligned to it. The transformations 
calculated for the chosen alignment axis are propagated to the remaining axes, saving time and avoiding 
disparities between the alignments of different dimensions. For the transformation itself, the pixel 
values are calculated from the symmetric cubic uniform B-spline representation of the original image. 
Internally, the images are first normalized to a range of approx. \-1 to 1 and later rescaled to the original 
scale. The data is converted back to the image pixel type (byte, short, int, ...) and the original 
image data is overwritten. This approach leads to a number of limitations:

* All images in the stack MUST have the same size.
* Discontinuous or tiled image planes are not supported at the moment.
* Only three dimensions are currently supported (C, Z, T)

Furthermore, these operations will add numerical inaccuracies to the pixel values. In case the raw values are 
needed for quantitative image analyses, the applied transformations can be exported to a file, from which the 
pixel location in the original image can be calculated. It is recommended to use this approach when 
performing statistical analyses or quantifications using the pixel intensity values.

The empty image regions are zero filled. Because the current version does not support masking of image regions 
this leads to an important caveat. Aligned stacks cannot be realigned along another axis, since the zero filled 
pixels will contribute to the error function and thus influence the alignment. Similarly, images of different 
sizes cannot be padded with zeros as this will also influence the alignment. The only multi-axis alignment 
currently implemented is Z -> T. In case more complex alignments are needed, the axes have to be registered 
separately and the transformation parameters have to be exported. These can then be combined externally and the 
images can be transformed with the combined transformations. Currently it is not possible to import the 
transformations and apply them to the stack.

There are different internal modes of operation. The image registration algorithm has been implemented for the 
CPU as plain Java and for GPUs in OpenCL. Both implementations support double-precision and hybrid-precision modes. 
Double precision is only used for images of type double or long/unsigned long. For other 
datatypes the hybrid precision mode is used. In hybrid precision mode the image pyramids, coefficient pyramids 
and all other required helper structures (such as the derivative pyramids) are calculated in single precision 
floating point, because this is much faster on GPUs and requires less memory. This also enabled the use of 
certain optimizations described by [Ruijters et al. (2012)](https://doi.org/10.1093/comjnl/bxq086). The results 
from the single precision stage are then used as optimal initial parameters for the final stage of the hybrid 
precision mode, which uses double precision floating point numbers to reduce the errors as much as possible. 
For cases where single precision floating point numbers are sufficiently accurate, there is a GPU single 
precision only mode. For normal imaging with \<= 16 bits this mode is usually sufficient and quite fast. Although 
the plugin was designed to run registration in parallel on both GPUs and CPUs for optimal 
performance, it turns out that cross-synchronization and resource bottlenecks often make this approach slower 
than the GPU only modes. Note that because of certain platform specific implementations (such as the cascaded 
parallel tree sum reduction) the CPU and the GPU code will never produce exactly the same results. In case a 
more reproducible registration is required it should be restricted to only CPU or only GPU.
Please note that the number of cores in modern CPUs has increased significantly since the inception 
of this project. Because the implementation will use all available cores when using one of the CPU 
alignment modes, this may lead to memory exhaustion and render the system inoperable. This happens 
because all of the buffers (such as derivative and B-spline coefficient pyramids) are allocated per core. 
In a future version the available memory may be taken into account.

Because the optimization criterion is minimization of the global MSQE, the software typically yields the best results 
for Z-stacks or time series with small distances or timesteps respectively. In case the background contains
features which could dominate the error function, the optimization will not yield satisfactory results.

The project relies heavily on [JogAmp JOCL](https://jogamp.org/jocl/www/) which may conflict with CLIJ/CLIJ2 and other
tools which require JOCL. Unfortunately testing on Ubuntu with an NVIDIA graphics card gave completely irreproducible
NaN errors when running kernels with mixed single and double precision floating point operations. Thus, the OpenCL accelerated
version is currently only available on Windows(TM). It is possible to run the plugin on the CPU when JOCL is not available
and the switch to the CPU will be silent.

In lieu of a publication describing the development of NGStackReg please either cite its first use:

Ringel, P.D., Di Hu, and Basler, M. (2017). The Role of Type VI Secretion System Effectors in Target Cell Lysis and Subsequent Horizontal Gene Transfer. *Cell reports* **21**:3927-3940 (DOI: [10.1016/j.celrep.2017.12.020](https://doi.org/10.1016/j.celrep.2017.12.020)).

or, if you prefer to cite the actual description of the plugin development, please cite my PhD thesis:

Ringel (2018) Mechanisms of delivery and mode of action of type VI secretion system effectors. *Doctoral Thesis* (DOI: [10.5451/unibas-006805400](https://doi.org/10.5451/unibas-006805400))


## Release notes for version 0.2.0

* Fixed boundary condition miscalculations of the interpolation indices of a number of OpenCL kernels (later refactored for single point of error)
* Fixed a number of spelling and reference mistakes in the exported transformations
* Fixed erroneous alignment mode changes when a GPU is not available
* Added support for scaled rotation transformations (rotation with scaling and translation).
* Added support for affine transformations (translation, rotation, scaling and shearing).
* Added canvas resizing as an option. This resizes the canvas of the aligned images such that the entire aligned stack fits into the canvas. This is especially useful for large rotations and shearing transformations.
* Added SPIR-V (64-bit) IL compiled binaries for the OpenCL programs to speed up the kernel loading and compilation time.
* Refactored OpenCL code to reduce some of the code duplication
* Changed the OpenCL code to make explicit use of vectorized operations:
    - using dedicated vector and scalar math functions (dot, fma) to accelerate and consolidate some calculations (in part this also optimized some of the memory access)
* Refactored the Java code to reduce some of the code duplication and to make it more modular and maintainable. This also included a refactor of the transformation export code to make it more robust and easier to extend in the future.

## Known limitations

* The current method of passing a file to save the transformations to is not very conducive to scripting.
* All images in the stack MUST have the same size.
* Only three dimensions are currently supported (C, Z, T)
* The minimum supported image size is 24x24 pixels.
* Discontinuous or tiled image planes are not supported at the moment.
* Masking of image regions is not supported.
* Volume (3D) alignment is not supported.
* The current export format for the transformations is a custom JSON format but this may change in the future.
* Transformations cannot be imported and applied to a stack.
* ImageJ/FIJI can sometimes assign the Z axis to the time axis. When running an alignment with the axes mixed up, this may result in unanticipated behavior.
* When aligning an already aligned stack along another axis, the black pixels on the border of the aligned images will cause the second alignment to produce non-optimal results. This is because the black pixels will skew the error calculation between the images. Use the multidimensional alignment mode.
* On Ubuntu using NVIDIA drivers resulted in irreproducible results and random NaNs using the hybrid precision OpenCL code. Because the root cause could not yet be identified the OpenCL acceleration is completely disabled on all platforms except for Windows on which it seems to work just fine. 

## TODO:

* [x] Add scaled rotation transformation
* [x] Add affine transformation
* [ ] Add import of transformations (or use TrackEM2 xml format)
* [ ] Maybe: Add masking
* [x] Maybe: Add image resizing
* [ ] Maybe: Add ops or service functionality

## References:

1. Ringel (2018) Mechanisms of delivery and mode of action of type VI secretion system effectors. *Doctoral Thesis* (DOI: [10.5451/unibas-006805400](https://doi.org/10.5451/unibas-006805400))
2. Ruijters D &amp; Th&eacute;venaz P (2012) GPU Prefilter for Accurate Cubic B-spline Interpolation. *The Computer Journal* **55**:15–20 (DOI: [10.1093/comjnl/bxq086](https://doi.org/10.1093/comjnl/bxq086))
3. Th&eacute;venaz P, Ruttimann UE &amp; Unser M (1998) A pyramid approach to subpixel registration based on intensity. *IEEE transactions on image processing : a publication of the IEEE Signal Processing Society* **7**:27–41 (DOI: [10.1109/83.650848](http://dx.doi.org/10.1109/83.650848))
4. Unser M (1999) Splines:A perfect fit for signal and image processing. *IEEE Signal Process. Mag.* **16**: 22–38 (DOI: [10.1109/79.799930](http://dx.doi.org/10.1109/79.799930))
5. Unser M, Aldroubi A &amp; Eden M (1993) B-spline signal processing: Part I - Theory. *IEEE Trans. Signal Process.* **41**: 821–833 (DOI: [10.1109/78.193220](http://dx.doi.org/10.1109/78.193220))
6. Unser M, Aldroubi A &amp; Eden M (1993) B-spline signal processing: Part II - Efficiency design and applications. *IEEE Trans. Signal Process.* **41**: 834–848 (DOI: [10.1109/78.193221](http://dx.doi.org/10.1109/78.193221))
