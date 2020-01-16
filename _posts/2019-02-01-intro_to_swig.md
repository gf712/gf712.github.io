---
layout: post
title:  "Extending C++ with SWIG"
categories: [programming]
tags: [shogun, C++, Python, SWIG]
---

## Introduction
Shogun's core library is written in C++, that is everything from memory management, exception hadling to the linear algebra framework that is required to write the machine learning algorithms. However, C++ is not the laguage of choice of data scientists and even machine learning engineers. This is despite the large effort that has been made in modern C++ to make memory management something almost from the past (use `std::shared_ptr` instead) and making types automatically deduced, i.e. with `auto`. Most scientist do know how to write Python, statisticians in particular usually know R, most engineers prefer Java, and possibly C#, and then languages such as Go are becoming more relevant. It is no coincidence that these languages, and more, are covered by SWIG. 

## SWIG
SWIG, or Simplified Wrapper and Interface Generator, is a tool that uses a target language's C-API to call C/C++ functions. The SWIG parser reads files with a `.i` extension and generates a translation unit with all the function calls that are exposed to the target language and calls to function that do the respective type conversions. Standard library containers, such as `std::vector` are supported in most languages for example, and in Python are represented as sets. That means when a C++ function that returns a `std::vector` is called from the source code created by SWIG it returns a Python `set`. However, you might need a `list` instead, and this can be done with some ease in SWIG. For this you require so called typemaps, that map C/C++ types to the target language's type. The conversion is done with a user provided function, and has to name a specific type, as opposed to templated type. For example you need a typemap for `std::vector<int>`, `std::vector<unsigned int>`, `std::vector<float>` and so on. 
```cpp
%typemap(out) std::vector<int>
	$result = PyList_New($1.size);
	for (int i = 0; i < $1.size; ++i)
	{
		int py_value = PyLong_FromLong($1[i]);
		int result_code = PyList_SetItem($result, i, py_value);
		if (result_code == -1)
		{
			PyErr_SetString(PyExc_RuntimeError, 
				"Failed to create list from std::vector<int>.");
		}
	}
```
This function is pretty self explanatory, but there are a couple of special keywords/variables here. First, this block of code will be copy pasted by the SWIG parser in the translation unit after each function call that returns `std::vector<int>`. The `$result` variable will be replaced by the appropriate name inside the function body, and has type `PyObject*`, and will be also the return value of the Python wrapper function. And then `$1` is the result of the call to the wrapped function, so we can for example get the size of the allocated vector with `std::vector::size`, i.e. `$1.size()`.

SWIG provides a whole range of customisable behaviours that might need adjustment for each target language, too many to be covered here. However, another important feature worth mentioning, that is used in Shogun, is the reference counter. Currently, in Shogun most objects have a reference counter that simply increments and decrements when a new lvalue references a given object. This is a similar behaviour to Python's `Py_INCREF` and `Py_DECREF` (`SG_REF` and `SG_UNREF`, respectively, in shogun). This old school system is now mostly obsolute in modern C++ since smart pointers have been introduced in the standard, but a lot of code still depends on manual counting. And SWIG extends this to the target languages with `%ref` and `%unref`. This means that when a value is assigned in Python the behaviour bound to `%ref`, i.e. SG_REF, is called.
```python
sg.kernel("GaussianKernel") # deleted immediately as the refcount is always 0
a = sg.kernel("GaussianKernel") # refcount is 1 (assignment to `a` increases it to 1)
b = a # refcount is 2

def func(c)
   local_var = c
   ...
   return result

c = func(a) # refcount increases to 3 and goes back to 2 when local_var goes out of scope
```
When this short program terminates all variables are garbage collected, the reference counter goes to 0 and the Shogun object's destructor is called. No memory leaks!

SWIG takes care of a lot of boiler plate code for various languages, and is a truly amazing developer tool for projects that require various interfaces like Shogun. It is also being actively developed and is compliant with most C++ code that you'll need to expose to a user, e.g. constructors, function calls and getters.

Checkout my post on how to [extend C++ to Python with SWIG with typed getters]({% post_url 2019-03-01-extend_python_getters %}) for some more advanced functionality!