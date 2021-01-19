# Athena: the BCI architecture library
Athena is a library that comprises many different bci frameworks that perform classification on a set of eeg data. It contains methods in the same philosophy as the Keras layers to construct BCI architectures to:
* Load data
* CSP filters
* Classifier training
* Decission making.

It contains different versions of the Enhanced Multimodal Fusion framework and the Traditional Framework in [1] already implemented using the blocks from this library.

Besides the basic methods for constructing the different architectures, different plugins regarding different variations and ampliations of the basic functionality are also presented, mostly related to further developments in the Decission making phase. Developing additional plugins is easy as long as the same philosophy of work is followed.

As of today, the best result for the frameworks here obtained in

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

[1] Fumanal-Idocin, J., Wang, Y., Lin, C., Fern'andez, J., Sanz, J.A., & Bustince, H. (2021). Motor-Imagery-Based Brain Computer Interface using Signal Derivation and Aggregation Functions.