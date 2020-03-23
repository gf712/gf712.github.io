---
layout: post
title:  "Graphs are not everyone's cup of tea"
categories: [programming]
tags: [shogun, C++, Computational graphs]
---

In a [previous post]({% post_url 2020-03-17-shogun_graphs %}) I introduced the work that we have done at Shogun to develop a computational graph backend for our linalg module. The graph abstracts away all the computations and when built/compiled it does all the optimisations, such as merging ops and allocations, and is ready to receive data. The problem with this approach is that the user is exposed to a two-step process, build and evaluate, rather than a single step execution that most of us are used to with linear algebra libraries. The former method was largely popularised in TensorFlow 1.x (inspired by Theano), and the latter started appearing in later TensorFlow versions with [eager execution](https://www.tensorflow.org/guide/eager) and now with [JAX](https://github.com/google/jax). I am not familiar with all the details, but from my understanding, eager execution forces each expression to be evaluated immediately. So, for example, in `y = X.dot(w) + b` I would get the value of `y` after executing the right-hand side expression. In JAX however, `y` is a lazy expression and it will only be calculated until the very last moment. Execution would be triggered by serialisation, so when doing something like `print(y)`, or looping through the values. The JAX approach can fully leverage the advantages of running graphs, because computation is deferred until after we declared all the calculations we want to perform. In my opinion, this is the better tradeoff between user friendliness and efficiency.

# Where does the data come from?
The beauty, in some sense, of the computational graph is that when we provide some data we are either copying it to the graph or moving ownership (or creating a view to it). This means there is a strong guarantee that the graph will get a valid chunk of memory and will not `segfault` on us during execution (assuming for now that evaluation is not asynchronous). This guarantee falls apart when using a JAX-style approach. Instantiating a `jax.numpy.array` like class with a data view (like [`view`](https://en.cppreference.com/w/cpp/ranges/view) in C++ ranges) does not guarantee that the data will be around until execution, unless we copy it (which might kill performance with a large batch size or high dimensional data) or do an explicit move. Of course, the copy will be performed if we are using a `GPU` or any other `device`, but if we are using just the `host` CPU, which would be the most common setup in prototyping and small model training, we are performing unnecessary copies.
```cpp
Array computation_1()
{
    SGVector X(100'000);
    X.range_fill();
    // create an array with a view of the data in X
    auto array = Array::create_view(X);
    auto result = array + array;
    // before exiting I have to do something with X
    // and that is why I didn't move it
    std::cout << X << '\n';
    // returns lazy expression
    return result;   
}

void main()
{
    // execution happens in `main` rather than computation_1
    std::cout << computation_1() << '\n';
}
```
In the code snippet above we will segfault, as the destructor of `X` (`~SGVector`) will deallocate the internal data pointer and set it  to `nullptr` and when evaluating the lazy expressions there will be no data. You might say that we could have moved the data, but then `X` would have been in a "valid but undefined state", so I wouldn't be able to print its contents anymore. This is definitely not a major deal breaker, but it is something to keep in mind when moving to this type of functional API.

# Lazy expressions
I must admit, when thinking about the implementation of `Array` I had to spend some time running arbitrary code with JAX within `pdb`. So, the design of `LazyExpr` might look familiar if you know how JAX works internally.
`Array` is the entry point, it is the equivalent of `jax.numpy.array` and it performs a `device_put` or `device_view` operation to provide access to the data. Behind the scenes it starts building the graph too, it will actually instantiate a `node::Input` which you will have seen in my [previous post]({% post_url 2020-03-17-shogun_graphs %}). In order to abstract away the user facing API (`Array`) and this graph building process, there is an additional class: `LazyExpr`. Each `Array` instance has a unique `LazyExpr` pointer (using `std::unique_ptr` of course) that tells the `Array` what to do when it needs to be evaluated.
``` cpp
auto prediction_part1 = X.dot(w);
auto prediction_part2 = prediction_part1 + b;
```
In this expression we create an `Array` for `X.dot(w)` with it's own `LazyExpr` and then another when adding `b`. Because they own an instance of `LazyExpr`, where the second one is built with a copy of the first, we can evaluate these `Array`s independently. If we only had one `LazyExpr` we would lose track of what each `Array` owns. So if I were to evaluate `prediction_part1` I would get the result of `prediction_part2`.

When performing an operation on `Array` we bind the right-hand side to the left-hand side using the abstract representation of this operation (a `Node`). So the implementation of addition looks like this:
```cpp
std::shared_ptr<Array> operator+(
    const std::shared_ptr& lhs,
    const std::shared_ptr& rhs)
{
    auto expr = lhs->get_lazy_expression()->copy();
    expr->bind<node::Add>(rhs->get_lazy_expression());
    return std::make_shared(std::move(expr));
}
```
So here we copy the `LazyExpr` of `lhs` and then call `bind` on it using `node::Add`, which is the abstract representation of the addition operation. This is the copy that is then passed on to the new `Array` instance and ensures that each expression is only owned by a single `Array`. `bind` itself is a variadic class member function of `LazyExpr` that can handle all operations:
```cpp
template <
    typename OperatorType, typename... Args,
    std::enable_if_t<std::is_base_of_v<node::Node, OperatorType>>* =
        nullptr>
void bind(Args&&... args)
{
    (bind_input(std::forward(args)), ...);

    m_output = std::make_shared<OperatorType>(
        m_output, return_node(std::forward(args))...);
}
```
The class function `LazyExpr::bind_input` binds the right-hand side `LazyExpr`s to `LazyExpr::m_inputs`, which tracks the input nodes for each expression. These inputs are then passed to the `Graph` instantiation, later on.
```cpp
template <typename T>
void bind_input(const T& expr)
{
    if constexpr (std::is_same_v>)
    {
        for (const auto& [input_node, input_tensor] :
             expr->get_inputs())
        {
            if (!m_inputs.count(input_node))
            {
                m_inputs.emplace(input_node, input_tensor);
            }
        }
    }
}
```
In the last line of `LazyExpr::bind` we essentially have `m_output = std::make_shared<node::Add>(m_output, rhs);` in the case of `operator+`. Simple, yet efficient, using some C++17 features, as we are now stamping out the implementation of each `Node`'s respective `LazyExpr::bind` at compile time!
When the `Array` is evaluated it can now get the inputs of the `LazyExpr` and its output (now each operation has a single output). This is passed on to the `Graph`, which is then built, and the `Array` evaluation can pass the input data (which was also stored in `LazyExpr`). In the near future we will have implemented graph caching so that this API works well in loops:
```cpp
for (const auto& X: data_iterator)
{
    // cache the internal graph here
    auto y = X.dot(y) + b;
    serialize(y);
}
```

# Conclusion
In a nutshell, we moved away from a graph declaration API to a functional API which resembles NumPy (and JAX), but still uses computational graphs.

```cpp
// part 1: declare the graph
auto input1 = make_shared<node::Input>(Shape{Dynamic}, element_type::FLOAT64);
auto input2 = make_shared<node::Input>(Shape{Dynamic}, element_type::FLOAT64);
auto output = input1 + input2;
graph = make_shared<Graph>({input1, input2}, output);
graph->build(GRAPH_BACKEND::NGRAPH);
// part 2: evaluate graph
auto result = graph->evaluate(vector{make_shared<Tensor>(X1), make_shared<Tensor>(X2)});
std::cout << result[0];
```
becomes:
```cpp
auto input1 = make_shared<Array>(X1);
auto input2 = make_shared<Array>(X2);
auto output = input1 + input2;
// choose backend before forcing evaluation (optional)
ShogunEnv::instance()->set_graph_backend(GRAPH_BACKEND::NGRAPH);
std::cout << output;
```

The latter definitely looks more user friendly! In a future post I will cover graph caching and hopefully autodifferentiation, once that is implemented!