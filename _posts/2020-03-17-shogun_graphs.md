---
layout: post
title:  "Building a computational graph backend for Shogun"
categories: [programming]
tags: [shogun, C++, Computational graphs]
---

At Shogun we try to come up with very efficient implementations of machine learning algorithms, such as SVMs and other kernel methods. The original approach was to write code with lots of calls to BLAS and LAPACK, sprinkled across the code base. These were then replaced with Eigen calls, which eventually were hidden behind the `linalg` module. However, things are changing fast in the machine learning world, both in terms of hardware and software. Nowadays, functionalities such as autodifferentiation are frequently a requirement when working with kernels, see for example [GPFlow](https://www.gpflow.org/). However, in Shogun we still rely on manual implementations of gradient calculations, which is both error prone and time consuming. In addition to that, most research setups have heterogeneous hardware, e.g. CPU+GPU/TPU or distributed clusters, and determining where to run an algorithm at compile time is not realistic, or practical. For example, if I want to run a DNN I want to have the control at runtime where it will run, i.e. whether to use a GPU or a cluster. In Shogun, we currently have some support for GPU using [ViennaCL](http://viennacl.sourceforge.net/), but it is mostly outdated and requires us to keep track of an additional dependency. In conclusion, the machine learning requirements and available resources are very different now compared to when Shogun started in 1999!

## A brief history of linalg
As mentioned above, there has been a constant evolution in Shogun around how to perform linear algebra computations. The objective has remained the same though: have an efficient implementation that is developer/user friendly, but also reasonable to compile, e.g. doesn't take up all your memory when compiling in debug mode. The `linalg` module is very user friendly and is a runtime abstraction of Eigen. It hides away implementation details and can handle the most common linear algebra calls, such as matrix matrix multiplication and even matrix decomposition, and could technically be exposed to shogun users using one of the interface languages, such as Python. This is because `linalg` is compiled as a whole and is exported in the dll, rather than being header only and each computation recompiled for each algorithm. 

This seems fine so far, but the issue is that [Eigen](https://eigen.tuxfamily.org/) is really good at detecting patterns at compile time and optimise [expression templates](https://en.wikipedia.org/wiki/Expression_templates). But each `linalg` function hides the implementation and therefore there is no compile time optimisation when combining `linalg` calls. The other issue is that it will be compiled for your hardware specifically, so it can be a distribution nightmare (imagine having to maintain distributions for all possible CPU/GPU architectures...). And lastly, and maybe the most important point, `linalg` doesn't support autodifferentiation, and was not designed for this purpose.

## Moving towards computational graphs
In the machine learning world computational graphs are just fancy named abstract semantic graph (ASGs, an abstraction of ASTs) with some domain specific functionality. In a nutshell they translate complex computations, let's say a DNN, into a series of dependent expression nodes that represent elementary computations, such as addition. The graph has some specific properties, because it is actually a DAG, but that is not important for now (it will be covered in future posts about optimisation and scheduling).

The graph data structure in itself is not difficult to implement. It is made of various nodes that are connected to each other, without creating any cyclic dependencies, which maps input nodes to output nodes.

So as a starting point I implemented, together with [Viktor](https://github.com/vigsterkr) (another Shogun dev), a simple `Graph` abstraction. `Graph` builds a DAG using input and output nodes. Each `Node` has information about the children `Node`s, so the graph can traverse the dependent nodes, from output to input. The API we came up with is very heavily inspired by the `NGraph` implementation, which is also similar to `Tensorflow 1.x`.

```cpp
auto input1 = make_shared<node::Input>(Shape{Shape::Dynamic}, element_type::FLOAT32);
auto input2 = make_shared<node::Input>(Shape{10}, element_type::FLOAT32);

auto intermediate = input1 + input2;

auto output = intermediate + input1;

auto graph = make_shared<Graph>(
    vector{input1, input2},
    vector<shared_ptr<node::Node>>{intermediate, output});
```

In this example the `Graph` is aware of two ouputs: `intermediate` and `output` and knows that there two inputs: `input1` and `input2`. We also declare that we don't know the shape of `input1`, we only know it's a vector, i.e. 1D tensor. The `Graph` now knows exactly what the user wants and can perform all sorts of optimisations, such as reduce number of allocations and fuse operations (in this case very little can be optimised). This step is performed when `graph->build()` is called. 

Something that sets our implementation apart is that we wrote the code in order to support various graph implementations at **runtime**. This allows the user to choose what he/she thinks is the best implementation. For example, Shogun's lightweight graph might be better suited for very simple calculations, as other implementations add a large overhead, due to the large number of optimisations they can perform. So far we only support `NGraph`, but there are plans to add `XLA`.

The next step is to pass the data to the graph so the outputs can be calculated. This is also when dynamic shapes are "frozen", as the input will determine their shapes.
```cpp
auto result =graph->evaluate(vector{make_shared<Tensor>(X1),make_shared<Tensor>(X2)});

auto result1 = result[0]->as<SGVector<float32_t>>();
auto result2 = result[1]->as<SGVector<float32_t>>();
```

The current implementation is Shogun specific, i.e. the data from Tensor is meant to be passed on to a `SGVector`/`SGMatrix` using `Tensor::as<T>()`, but this will be changed in the future as we move towards a standalone implementation. Furthermore, the current implementation of `Graph::evaluate` reevaluates the whole graph, even when parts of the graph have already been computed. We are currently working on caching data to be accessed again easily. This would be useful if for some reason I were to build the graph first with `intermediate` as an output, and later build another graph using `output` instead.

Another feature we are developing at Shogun is a JAX style syntax, i.e. there will be a functional API, where the graph is hidden away from the user, and resembles what Eigen achieves at compile time with expression templates.
```cpp
auto X1 = SGVector<NumericType>(10);
auto X2 = SGVector<NumericType>(10);

auto input1 = make_shared<Array>(X1);
auto input2 = make_shared<Array>(X2);
auto intermediate = input1 + input2; // no evaluation
auto output = intermediate + input1; // no evaluation

std::cout << output << '\n'; // forces evaluation using input1 and input2
```
However, no matter what API the user chooses to use, the graph is built in the same way and fetches the runtime implementations depending on the chosen backend, which might provide various optimisations such as JIT, MLIR, GPU/TPU implementations and so on.

## Conclusion
Computational graphs provide a nice abstraction that allows for runtime optimisations, depending on the software and hardware available. They also facilitate gradient calculations using forward and reverse mode autodifferentiation, which is not implemented in Shogun yet, but will be soon. What I told you in this post was largely theoretical, but in the near future I will write a post with some benchmarks and show how computational graphs could be used in the new Shogun `linalg` module. I will also cover how we are implementing a JAX like API in C++, what optimisation we can come up with and how we managed to lookup implementations at runtime! For "live" developments checkout the [modular-graph branch on Shogun's repo](https://github.com/shogun-toolbox/shogun/tree/feature/modular-graph)!

## PS
Here is also a simple demo using xeus-cling: 
{% gist e30a413ef7a91dae5d8fe0235e07af29 %}