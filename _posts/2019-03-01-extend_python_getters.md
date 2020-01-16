---
layout: post
title:  "Type agnostic Python getters"
categories: [programming]
tags: [shogun, C++, Python, SWIG]
---

## Introduction
C++ is a statically typed language, meaning that at compile time the types of each variable are checked and the validity of operations with these variables is asserted. On the other hand, Python is dynamically typed, and a variable can hold any type.
This leads to an awkward syntax in shogun where the getters need to know the return type at runtime. In other words, if a class member is a float we need to use the `get_real` getter, if it is an int we would use `get_int`. 
```python
>>> import shogun as sg
>>> lr = sg.machine("LibLinearRegression")
>>> lr.get_int("max_iterations")
>>> # this will raise an error
>>> lr.get_real("max_iterations")
```

## Extend Shogun to dynamically typed languages
This is an issue when extending shogun to OpenML, because we need to use these getters constantly and we would have to keep track of these getters in a third-party application. For example, we would have to map class member names to their types, and that would be intractable.
Instead we decided to use a `try: ... except: ...` block inside a loop. The loop iterates over a list of all getters, e.g. `["get_int", "get_real", ...]`, and tries to retrieve a class member with different types until no error is returned. This is of course very innefficient but we argue that this part of the machine learning pipeline is very short in time, relative to the whole process.
So now we can use something as simple as:
```python
>>> import shogun as sg
>>> lr = sg.machine("LibLinearRegression")
>>> lr.get("max_iterations")
```
## The details
In this section I will briefly go into more implementation details that are not directly related to the problem, but show how SWIG and C++ can be very powerful together (and dangerous for the end user).

### Monkey patch SWIG (using Python's C-API)
Before I wrote the new `get` behaviour in shogun I realised I had to append a method to SGObject, which is the C++ class from which all other shogun classes are inherited from. In Python almost everything is possible. This is because every `PyObject` (in CPython) is a `struct` which holds the data and all the `PyObject`s sit all in a class or module (sort of a namespace in C++). Unfortunatelly using pure Python code you cannot just add a new function to class. For example you can't do this:
```python
>>> class A:
      def methodA(self):
        return 42
# very wrong code
>>> A["methodB"] = lambda x: 43
>>> a = A()
>>> a.methodB()
```
This is because types are immutable, which makes sense in this case. If this wasn't the case anyone could add new methods to a class and break the code.
To add behaviour to a class we can just inherit from it, so in this case we could do this:
```python
>>> class B(A):
      def methodB(self):
        return 43
>>> a = B()
>>> a.methodA()
>>> a.methodB()
```
Even though this is valid Python code, this is not useful in this case because all shogun objects inherit from SGObject and we can't just change this inheritance that easily to use a modified SGObject class.
However, this logic can be implement in C/C++ using the CPython C-API, i.e. add a new key/value (function name/ function) pair to `class A`. After some searching around the web I found a monkey patch solution suggested in a SWIG [issue](https://github.com/swig/swig/issues/723#issuecomment-230178855).
```cpp
%typemap(out) void _swig_monkey_patch 
"$result = PyErr_Occurred() ? NULL : SWIG_Py_Void();"
%inline %{
	static void _swig_monkey_patch(PyObject *type, PyObject *name, 
		PyObject *object) {
		PyObject *dict = NULL;
#if PY_VERSION_HEX>=0x03000000
		if (!PyUnicode_Check(name))
#else
		if (!PyString_Check(name))
#endif
			{
				PyErr_SetString(PyExc_TypeError, 
					"name is not a string");
				return;
			}

		if (PyType_Check(type)) {
			PyTypeObject *pytype = (PyTypeObject *)type;
			dict = pytype->tp_dict;
		}
		else if (PyModule_Check(type)) {
			dict = PyModule_GetDict(type);
		}
		else {
			PyErr_SetString(PyExc_TypeError, 
				"type is not a Python type or module");
			return;
		}
		if (PyDict_Contains(dict, name))
		{
			PyErr_SetString(PyExc_ValueError, 
				"function name already exists in the "
				"given scope");
			return;
		}
		PyDict_SetItem(dict, name, object);

	  }
%}
```
This is a slightly modified version of the code provided in the linked issue in order to also add a new function to a Python module (`dict = PyModule_GetDict(type);`) and some additional checks, just in case at some point someone else uses this function and starts changing things by accident.
Just a quick overview of this code snippet (I will skip the SWIG details):
 - Check if the name of the new function to be added to the class/module is a Python string
```cpp
#if PY_VERSION_HEX>=0x03000000
if (!PyUnicode_Check(name))
#else
if (!PyString_Check(name))
#endif
	{
		PyErr_SetString(PyExc_TypeError, 
			"name is not a string");
		return;
	}
```
- Check if the given scope (`type`) is a Python type, i.e. `class`, or a `module`, and get the dictionary with all the `PyObject`s in this scope
```cpp
if (PyType_Check(type)) {
	PyTypeObject *pytype = (PyTypeObject *)type;
	dict = pytype->tp_dict;
}
else if (PyModule_Check(type)) {
	dict = PyModule_GetDict(type);
}
else {
	PyErr_SetString(PyExc_TypeError, 
		"type is not a Python type or module");
	return;
}
```
- Check that the new function isn't shadowing/overriding an existing function:
```cpp
if (PyDict_Contains(dict, name))
{
	PyErr_SetString(PyExc_ValueError, 
		"function name already exists in the "
		"given scope");
	return;
}
``` 
- Update the dictionary of `PyObject`s `dict`:
```cpp
PyDict_SetItem(dict, name, object);
```

Now we can use this function to add a new `get` method to `SGObject` when the shogun module is imported. The new `get` method is just a function that checks the different getters `["get_int", "get_real", ...]` at runtime and returns the value of a model parameter if possible. It currently looks like this:
```python
def _internal_get_param(self, name):
    """
    Returns the value of the given parameter.
    The return type depends on the parameter,
    e.g. could be a builtin scalar or a
    numpy array representing a vector or matrix
    """

    for getter in _internal_getter_methods:
        try:
            return getter(self, name)
        except SystemError:
            pass
        except Exception:
            raise
    if name in self.parameter_names():
        raise ValueError("The current Python API does not "
        	"have a getter for '{}' of type '{}'".format(
        		name, self.parameter_type(name)))
    else:
        raise KeyError("There is no parameter called '{}' in "
        	"{}".format(name, self.get_name()))
```
And then we can just monkey patch `get` with this one liner:
```python
_swig_monkey_patch(SGObject, "get", _internal_get_param)
```
Now there is only one issue left: there is already a `get` method in `SGObject` (which returns values of type `SGObject`). So we need to rename this method to something different, for example `_get`. Additionally, the underscore in the new name is a Python convention to denote a private method (but this is not enforced by the interpreter).

### Renaming (Python) functions
There are two methods by which we can change the name of functions in this situation.
1. SWIG parser

There is a SWIG keyword to rename functions and variables called [`%rename`](http://www.swig.org/Doc3.0/SWIGDocumentation.html#SWIG_advanced_renaming). In our example `%rename(_get) get` would suffice. However, since we made `get` "private" we should also do the same with the other getters, i.e. `get_int` becomes `_get_int`, and so on. This starts becoming impractical, in particular when we want to just have this behaviour in Python.
2. Renaming Python functions with the C-API

Now that we know that a function in CPython is stored in a dictionary we can levarage this by renaming the key in this dictionary to whichever name we want the function to be known as. Again let's look at the final function and then break it down:
```cpp
%typemap(out) void _rename_python_function 
"$result = PyErr_Occurred() ? NULL : SWIG_Py_Void();"
%inline %{
static void _rename_python_function(PyObject *type, 
	PyObject *old_name, PyObject *new_name) {
	PyObject *dict = NULL,
			 *func_obj = NULL;
#if PY_VERSION_HEX>=0x03000000
	if (!PyUnicode_Check(old_name) || !PyUnicode_Check(new_name))
#else
	if (!PyString_Check(old_name) || !PyString_Check(new_name))
#endif
		{
			PyErr_SetString(PyExc_TypeError, 
				"'old_name' and 'new_name' have to "
                "be strings");
			return;
		}
	if (PyType_Check(type)) {
		PyTypeObject *pytype = (PyTypeObject *)type;
		dict = pytype->tp_dict;
		func_obj = PyDict_GetItem(dict, old_name);
		if (func_obj == NULL) {
			PyErr_SetString(PyExc_ValueError, 
				"'old_name' name does not exist in the "
                "given type");
			return;
		}
	}
	else if ( PyModule_Check(type)) {
		dict = PyModule_GetDict(type);
		func_obj = PyDict_GetItem(dict, old_name);
		if (func_obj == NULL) {
			PyErr_SetString(PyExc_ValueError, 
				"'old_name' does not exist in the given "
                "module");
			return;
		}
	}
	else {
		PyErr_SetString(PyExc_ValueError, 
			"'type' is neither a module or a Python type");
		return;
	}
	if (PyDict_Contains(dict, new_name))
	{
		PyErr_SetString(PyExc_ValueError, 
			"new_name already exists in the given scope");
		return;
	}
	PyDict_SetItem(dict, new_name, func_obj);
	PyDict_DelItem(dict, old_name);
}
%}
```

- Check if the arguments `new_name` and `old_name` are Python strings
```cpp
#if PY_VERSION_HEX>=0x03000000
if (!PyUnicode_Check(old_name) || !PyUnicode_Check(new_name))
#else
if (!PyString_Check(old_name) || !PyString_Check(new_name))
#endif
	{
		PyErr_SetString(PyExc_TypeError, 
			"'old_name' and 'new_name' have to be strings");
		return;
	}
```
- Get the `PyObject` pointing to our function (this is done slightly differently depending if we are dealing with a module or a class).
```cpp
if (PyType_Check(type)) {
	PyTypeObject *pytype = (PyTypeObject *)type;
	dict = pytype->tp_dict;
	func_obj = PyDict_GetItem(dict, old_name);
	if (func_obj == NULL) {
		PyErr_SetString(PyExc_ValueError, 
			"'old_name' name does not exist in the given type");
		return;
	}
}
else if ( PyModule_Check(type)) {
	dict = PyModule_GetDict(type);
	func_obj = PyDict_GetItem(dict, old_name);
	if (func_obj == NULL) {
		PyErr_SetString(PyExc_ValueError, 
			"'old_name' does not exist in the given module");
		return;
	}
}
else {
	PyErr_SetString(PyExc_ValueError, 
		"'type' is neither a module or a Python type");
	return;
}
```
- Add the new key/value pair, i.e. {`new_name`: `func_obj`}, and delete the original pair {`old_name`: `func_obj`}.
```cpp
PyDict_SetItem(dict, new_name, func_obj);
PyDict_DelItem(dict, old_name);
```

And that's it! Now we can loop through the function names we want to rename and add an underscore, and rename the function:
```python
_internal_getter_methods = []
for getter in _GETTERS:
    _private_getter = "_{}".format(getter)
    _rename_python_function(_shogun.SGObject, getter, _private_getter)
    _internal_getter_methods.append(
        _shogun.SGObject.__dict__[_private_getter])
```
Here `_GETTERS` is a list of all the getters in shogun that we want to expose to the interfaces, which is now the only list that has to be maintained as more getters are added. `_shogun.SGObject` is the SGObject Python class which is in the shared object `_shogun.so` compiled by SWIG, and `__dict__` accesses the [dictionary](https://docs.python.org/3/library/stdtypes.html#object.__dict__) of this class. Therefore `_shogun.SGObject.__dict__[_private_getter]` points to the current getter and is stored in the list `_internal_getter_methods` which is used in `_internal_get_param` (shown at the start of this post).

## Conclusion
In this post we saw what difficulties can arise at an API level when extending a statically typed language to a dynamically typed language. We then went over the solution that we implemented at Shogun to simplify the Python API and make it feel more pythonic by leveraging the CPython API and SWIG. This implementation has the advantage that it only has to be maintained in the library side and third-party extensions do not have to be updated as getters are modified.
Future plans include extending this to other dynamically typed languages used to interface Shogun!