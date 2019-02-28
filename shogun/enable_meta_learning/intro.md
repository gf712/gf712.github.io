## Enabling meta learning in Shogun - Introduction

### Project aim

This project aims to provide extensions to shogun for [Core ML](https://github.com/apple/coremltools) and [OpenML](https://www.openml.org). The philosophy behind both these open source projects is that machine learning projects should be easy to share across platforms. For example when I am collaborating with someone in a different institution I want them to have access to the same pipeline I have, that is the same algorithms from preprocessors to decision taking, and their parameters. This is the problem that OpenML tackles. Core ML focuses on the actual machine learning model and make it shareable across platforms. For example, I train a shogun SVM and then want to run some predictions on my IPhone. I could dedicate the next few days, weeks or even months learning Swift and trying to get an app working as an extension of shogun.. OR I can use Apple's Core ML which already is compatible with keras, scikit-learn and other popular frameworks to import models and respective parameters using Protobuf to an IPhone. This means I can also perform model training with keras on my laptop and then run inference in shogun.

### The challenge

This all sounds good, and somewhat straight forward. So what is the issue? It turns out that it can be a bit tricky to access class members in a type agnostic way when going from a strongly typed language like C++ to python. To understand this a bit better we need to look at how shogun interfaces C++ to other languages. The interfacing is done using [SWIG (Simplified Wrapper and Interface Generator)](http://www.swig.org), which wraps all the library code and extends it using the target language's C-API. For example to generate a Python library SWIG uses the [CPython API](https://docs.python.org/3.7/c-api/index.html).

### Work outline
