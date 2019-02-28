## Extending C++ with SWIG

Shogun's core library is written in C++, that is everything from memory management, exception hadling to the linear algebra framework that is required to write the machine learning algorithms. However, C++ is not the laguage of choice of data scientists and even machine learning engineers. This is despite the large effort that has been made in modern C++ to make memory management something almost from the past (use `std::shared_ptr` instead) and making types automatically deduced, i.e. with `auto`. Most scientist do know how to write Python, statisticians in particular usually know R, most engineers prefer Java, and possibly C#, and then languages such as Lua and Go are becoming more relevant. It is no coincidence that these languages, and more, are covered by SWIG. 

SWIG, or Simplified Wrapper and Interface Generator, is a tool that uses a target language's C-API to call C/C++ functions. The SWIG parser reads files with a `.i` extension and generates a translation unit with all the function calls that are exposed to the target language and calls to function that do the repective type conversions. Standard library containers, such as `std::vector` are supported in most languages for example, and in Python are represented as sets. That means when a C++ function that returns a `std::vector` is called from the source code created by SWIG and returns a Python `set`. However, you might need a `list` instead, and this can be done with some ease in SWIG. For this you require so called typemaps, that map C/C++ types to the target language's type. The conversion is done with a user provided function, and has to name a specific type, as opposed to templated type. For example you need a typemap for `std::vector<int>`, `std::vector<unsigned int>`, `std::vector<float>` and so on. 
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
This function is pretty self explanatory, but there are a couple of special keywords/variables here. First, this block of code will be copy pasted by the SWIG parser in the translation unit as is after each function call that returns `std::vector<int>`. The `$result` variable will be replaced to the appropriate name inside the function body, and has type `PyObject*`, and will be also the return value of the Python wrapper function. And then `$1` is the result of the wrapped function. 