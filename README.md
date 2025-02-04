# NGStackReg

NGStackReg provides a parallized and OpenCL accelerated complete rewrite of the StackReg/TurboReg plugins.

## TLDR

NGStackReg is a parallelized and OpenCL accelerated rewrite of the 
[StackReg](http://bigwww.epfl.ch/thevenaz/stackreg/) and [TurboReg](http://bigwww.epfl.ch/thevenaz/turboreg/) 
plugins which enables the registration of multidimensional image stacks. It was written with memory efficiency 
and registration speed in mind. Depending on the hardware quite significant speedups of >20x can be achieved 
compared to the original plugins. For images with <=16 bit depth GPU single precision mode is recommended 
otherwise GPU hybrid precision is recommended.

## Description

This plugin is meant to alleviate some inconveniences of the otherwise excellent 
[StackReg](http://bigwww.epfl.ch/thevenaz/stackreg/) and [TurboReg](http://bigwww.epfl.ch/thevenaz/turboreg/) 
plugins [(Th&eacute;venaz et al., 1998)](http://dx.doi.org/10.1109/83.650848). The inability of these plugins 
to align along a chosen axis poses a problem for typical multidimensional stacks, because each dimension needs 
to be aligned separately. Since [StackReg](http://bigwww.epfl.ch/thevenaz/stackreg/) passes each frame to 
[TurboReg](http://bigwww.epfl.ch/thevenaz/turboreg/) via a temporary image file with a hardcoded name, the 
alignments cannot be run in parallel, as already pointed out by the authors. In addition to the computational 
burden, aligning each axis separately may lead to disparities between the alignments. 
Some of these limitations have been addressed by previous projects like

PoorMan3Dreg (Liebling, 2010), 

[HyperStackReg (Sharma, 2015)](https://github.com/ved-sharma/HyperStackReg), 

[MultiStackRegistration (Busse and Miura, 2016)](https://github.com/miura/MultiStackRegistration) and 

[TimeLapseReg (Sahdev et al., 2017)](https://github.com/incfbelgiannode/TimeLapseReg). 

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

The plugin allows to choose the channel-, Z- or time-axis as alignment axis. Furthermore, the currently 
selected stack position is used as a reference for the alignment. In case an image in the center of 
the stack is of particular interest, just select it and everything will be aligned to it. The transformations 
calculated for the chosen alignment axis are propagated to the remaining axes, saving time and avoiding 
disparities between the alignments of different dimensions. For the transformation itself, the pixel 
values are calculated from the symmetric cubic uniform B-spline representation of the original image. 
Internally the images are first normalized to a range of approx. -1 to 1 and later rescaled to the original 
scale. The data is converted back to the image pixel type (byte, short, int, ...) and the original 
image data is overwritten. This approach leads to a number of limitations:

- All images in the stack MUST have the same size.
- Discontinuous or tiled image planes are not supported at the moment.
- The images are not resized to fit the transformation results.
- Only three dimensions are currently supported (C, Z, T)

Furthermore, these operations will add numerical inaccuracies to the pixel values. In case the raw values are 
needed for quantitative image analyses, the applied transformations can be exported to a file from which the 
pixel location in the original image file can be calculated. It is recommended to use this approach when 
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
CPU as plain Java and for GPUs in OpenCL. Both implementations may be run in double precision mode or in hybrid 
precision mode. Double precision is only used for images of type double or long/unsigned long. For other 
datatypes the hybrid precision mode is used. In hybrid precision mode the image pyramids, coefficient pyramids 
and all other required helper structures (such as the derivative pyramids) are calculated in single precision 
floating point, because this is much faster on GPUs and requires less memory. This also enabled the use of 
certain optimizations described by [Ruijters et al. (2012)](https://doi.org/10.1093/comjnl/bxq086). The results 
from the single precision stage are then used as optimal initial parameters for the final stage of the hybrid 
precision mode, which uses double precision floating point numbers to reduce the errors as much as possible. 
For cases where single precision floating point numbers are sufficiently accurate, there is a GPU single 
precision only mode. For normal imaging with <= 16 bits this mode is usually sufficient and quite fast. While 
the plugin was designed to run the registration process in parallel in both GPUs and CPUs for optimal 
performance it turns out that cross-synchronization and resource bottlenecks often make this approach slower 
than the GPU only modes. Note that because of certain platform specific implementations (such as the cascaded 
parallel tree sum reduction) the CPU and the GPU code will never produce exactly the same results. In case a 
more reproducible registration is required it should be restricted to only CPU or only GPU.

Because the minimization of the global MSQE is the optimization criterion, the software will yield
best results for Z-stacks or timeseries with small distances or timesteps respectively. In case the background contains
features which could dominate the error function, the optimization will not yield satisfactory results.

The project relies heavily on [JogAmp JOCL](https://jogamp.org/jocl/www/) which may conflict with CLIJ/CLIJ2 and other
tools which require JOCL. Unfortunately testing on Ubuntu with an NVidia graphics card gave completely irreproducible
NaN errors when running kernels with mixed single and double precision floating point operations. Thus, the OpenCL accelerated
version is currently only available on Windows(TM). It is possible to run the plugin on the CPU when JOCL is not available
and the switch to the CPU will be silent.

In lieu of a publication describing the development of NGStackReg please either cite its first use:

Ringel, P.D., Di Hu, and Basler, M. (2017). The Role of Type VI Secretion System Effectors in Target Cell Lysis and Subsequent Horizontal Gene Transfer. *Cell reports* **21**:3927-3940 (DOI: [10.1016/j.celrep.2017.12.020](https://doi.org/10.1016/j.celrep.2017.12.020)).

or, if you prefer to cite the actual description of the plugin development, please cite my PhD thesis:

Ringel (2018) Mechanisms of delivery and mode of action of type VI secretion system effectors. *Doctoral Thesis* (DOI: [10.5451/unibas-006805400](https://doi.org/10.5451/unibas-006805400))


## Known limitations:

- The current method of passing a file to save the transformations to is not very conducive to scripting.
- All images in the stack MUST have the same size.
- Only three dimensions are currently supported (C, Z, T)
- Currently only translation and rigid body transformations are supported.
- The stack is not resized to fit the transformed images.
- The minimum supported image size is 24x24 pixels.
- Discontinuous or tiled image planes are not supported at the moment.
- Masking of image regions is not supported.
- Volume (3D) alignment is not supported.
- The current export format for the transformations is a custom JSON format but this may change in the future.
- Transformations cannot be imported and applied to a stack.
- ImageJ/FIJI can sometimes assign the Z axis to the time axis. When running an alignment with the axes mixed up, this may result in unanticipated behavior.
- When aligning an already aligned stack along another axis, the black pixels on the border of the aligned images will cause the second alignment to produce non-optimal results. This is because the black pixels will skew the error calculation between the images. Use the multidimensional alignment mode.
- On Ubuntu using NVidia drivers resulted in irreproducible results and random NaNs using the hybrid precision OpenCL code. Because the root cause could not yet be identified the OpenCL acceleration is completely disabled on all platforms except for Windows on which it seems to work just fine. 

## TODO:

- [ ] Add scaled rotation transformation
- [ ] Add affine transformation
- [ ] Add import of transformations (or use TrackEM2 xml format)
- [ ] Maybe: Add masking
- [ ] Maybe: Add image resizing
- [ ] Maybe: Add ops or service functionality

## References:

1. Ringel (2018) Mechanisms of delivery and mode of action of type VI secretion system effectors. *Doctoral Thesis* (DOI: [10.5451/unibas-006805400](https://doi.org/10.5451/unibas-006805400))
2. Ruijters D &amp; Th&eacute;venaz P (2012) GPU Prefilter for Accurate Cubic B-spline Interpolation. *The Computer Journal* **55**:15–20 (DOI: [10.1093/comjnl/bxq086](https://doi.org/10.1093/comjnl/bxq086))
3. Th&eacute;venaz P, Ruttimann UE &amp; Unser M (1998) A pyramid approach to subpixel registration based on intensity. *IEEE transactions on image processing : a publication of the IEEE Signal Processing Society* **7**:27–41 (DOI: [10.1109/83.650848](http://dx.doi.org/10.1109/83.650848))
4. Unser M (1999) Splines:A perfect fit for signal and image processing. *IEEE Signal Process. Mag.* **16**: 22–38 (DOI: [10.1109/79.799930](http://dx.doi.org/10.1109/79.799930))
5. Unser M, Aldroubi A &amp; Eden M (1993) B-spline signal processing: Part I - Theory. *IEEE Trans. Signal Process.* **41**: 821–833 (DOI: [10.1109/78.193220](http://dx.doi.org/10.1109/78.193220))
6. Unser M, Aldroubi A &amp; Eden M (1993) B-spline signal processing: Part II - Efficiency design and applications. *IEEE Trans. Signal Process.* **41**: 834–848 (DOI: [10.1109/78.193221](http://dx.doi.org/10.1109/78.193221))
