
<p align="center">
  <img src="https://user-images.githubusercontent.com/12574757/105212305-90824d00-5b4d-11eb-982f-77a480075679.png">
</p>

# Athena: the BCI architecture library [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motor-imagery-based-brain-computer-interface/eeg-4-classes-on-bci-competition-iv-2a)](https://paperswithcode.com/sota/eeg-4-classes-on-bci-competition-iv-2a?p=motor-imagery-based-brain-computer-interface) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motor-imagery-based-brain-computer-interface/eeg-left-right-hand-on-bci-competition-iv-2a)](https://paperswithcode.com/sota/eeg-left-right-hand-on-bci-competition-iv-2a?p=motor-imagery-based-brain-computer-interface)
Athena is a library that comprises many different bci frameworks that perform classification on a set of eeg data. It contains methods in the same philosophy as the Keras layers to construct BCI architectures to:
* Load data
* CSP filters
* Classifier training
* Decission making.

It contains different versions of the Enhanced Multimodal Fusion framework and the Traditional Framework in [1] already implemented using the blocks from this library.

Besides the basic methods for constructing the different architectures, different plugins regarding different variations and ampliations of the basic functionality are also presented, mostly related to further developments in the Decission making phase. Developing additional plugins is easy as long as the same philosophy of work is followed.

As of today, the best result for the frameworks here obtained in this repo are reported in [1].

# Install

Just install this repo with the link:

``` pip install git+https://github.com/Fuminides/athena.git```


Or you an install it locally with pip:

```pip install .```



# Requirements

* Fancy_aggregations
* Numpy
* PyTorch (CCA plugin)
* Tensorflow (Sample processing plugin)

# Citation
If you use this work in any of your works, cite one of the following papers, preferably the first one:

[1] Fumanal-Idocin, J., Wang, Y., Lin, C., Fern'andez, J., Sanz, J.A., & Bustince, H. (2021). Motor-Imagery-Based Brain Computer Interface using Signal Derivation and Aggregation Functions.


[2] Fumanal-Idocin, J., Takáč, Z., Sanz, J. F. J. A., Goyena, H., Lin, C. T., Wang, Y. K., & Bustince, H. (2021). Interval-valued aggregation functions based on moderate deviations applied to Motor-Imagery-Based Brain Computer Interface. arXiv preprint arXiv:2011.09831.
