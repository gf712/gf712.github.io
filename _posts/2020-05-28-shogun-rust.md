---
layout: post
title:  "Machine learning with Rust: shogun-rust"
categories: [programming]
tags: [C++, Rust]
---

This year I decided to give [Rust](https://www.rust-lang.org/) a go. Rust is a low level compiled language that comes with various static analysis steps integrated into the compiler that prevents programmers from running into undefined/unexpected behaviour at runtime. Obviously, it is much more, but from a C++ programmer's perspective, this is what I see (and of course a lot of syntactic sugar and modern programming paradigms). At the start I felt like I was constantly battling the compiler, until I realised that maybe there is more than one approach to write code. And once this clicks, Rust's ownership and borrowing model becomes a powerful concept that allows you to write some code confidently, that *should* do what you expect at runtime. Note that this is not a tutorial on how to use Rust. In fact, I use Rust's [`unsafe`](https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html) keyword a lot (which switches off a lot of compiler checks), as it was the only way I found to expose a C++ library to Rust.

# Bringing C++ code to Rust

Now that I am a bit more confident in my Rust skills, I decided to create a [shogun-rust](https://github.com/gf712/shogun-rust) [crate](https://doc.rust-lang.org/book/ch07-01-packages-and-crates.html) that reuses Shogun's C++ library (I have not published it yet). Luckily, I am not the first person to write Rust code calling C++ libraries. However, there are several limitations in doing this, and it quickly became apparent to me that the OOP style in Shogun would be very difficult, or even impossible, to translate directly into Rust. Also I found that using C++ templates with Rust's generics became a bit awkward using [bindgen](https://github.com/rust-lang/rust-bindgen). As shogun is not a template heavy library and we make extensive use of type erasure to expose class members to the outside world, I decided to use a plain C interface that is called in the internals of Rust library. 

# A brief overview of the Shogun design
The Shogun class hierarchy is very simple. There is a base class, `SGObject`, and then various child classes are used for various machine learning related tasks. For example, `Machine` represents any algorithm that fits some data and performs inference based on learnt parameters (or data), or a `Transformer` applies some transformation to some features. All the child classes of these "interfaces" are not important, as the user will only be interested in the interface class functions, e.g. `Machine::train(const std::shared_ptr<Features>& X, const std::shared_ptr<Labels>& y)`. The parameters of the derived classes, e.g. number of bags in `RandomForest` (a `Machine`), are registered at runtime in a map which is owned by `SGObject`, and this is where we type erase the parameters, so that they are all in the same type agnostic map. The derived classes themselves are registered at build time in a C++ file, and can then be instantiated with a factory, e.g. `create<Machine>("RandomForest")`. 

# Crossing boundaries: C++ via C to Rust

One way to bring C++ to Rust is to use good old C. As much as I avoid writing C, it is really useful to glue together languages, as the ABI is far more "standard" (even though there is not standardised ABI), and all the types are well defined, e.g. no templates. Also using raw pointers, instead of C++'s smart pointers, can give you a lot more flexibility, in particular when calling C++ libraries from another language, which cannot control or access the reference count (easily, if at all).
I implemented Shogun's C++ library in C, by implementing various C functions that call the factories and the C++ class member functions. This also removed destructors, so each class has to have a free function that calls the destructor. The design is similar to what you would see in a OOP inspired C library such as [OpenSSL](https://github.com/openssl/openssl). I did avoid using "manual" vtables, as for the moment I did not require them.

The design is simple, and I just wanted to make it work, and it ended up doing exactly what I needed when it was processed by `bindgen`. Each "interface" class, e.g. `Machine` has a corresponding `create_` `C` function (`create_machine(const char* name)`). When calling the factory functions `C` gets a `sgobject_t` pointer, which in turn is implemented in C++. All `C` needs to know is that there is this pointer and keeping this address on the stack requires [`word`](https://en.wikipedia.org/wiki/Word_(computer_architecture)) size bytes. The implementation is completely abstracted away in C++. So in the background I make heavy use of modern C++ features, such as `std::variant` to store a `SGObject` derived class, i.e. `Machine`. I don't use the `SGObject` base class itself as I need to keep the type information, so I can see if I can call `train`, for example, without a `dynamic_cast`.
With this C-API I generated a `bindings.rs` file with `bindgen`, and then used Rust's [procedural macros](https://doc.rust-lang.org/reference/procedural-macros.html) to generate all the code necessary for each `SGObject` derived class:
```rust
#[derive(SGObject)]
pub struct Machine {
    ptr: *mut bindings::sgobject,
}
```
Which I can then call:
```rust
let rf = Machine::new("RandomForest");
match rf {
    Ok(_) => println!("All good"),
    Err(msg) => println!("ShogunException: {}", msg),
}
```
Without going into too many details, the procedural macro `SGObject` (the naming is a bit confusing) generates calls to the factories and destructors in the C API. It also handles calls to retrieve parameters from the underlying C++ `SGObject` and returns them as Rust's `Any` trait. The parameters can then be casted in Rust to the correct type.
```rust 
match gaussian.get("log_width") {
    Ok(value) => match value.downcast_ref::<f64>() {
        Some(fvalue) => println!("log_width: {}", fvalue),
        None => println!("log_width not of type f64"),
    },
    Err(msg) => println!("{ShogunException: }", msg),
}
``` 

# Handling exceptions
You might be aware that C++ exceptions are not supported in stable Rust, so I actually had to use a minor hack to get this working, otherwise a `ShogunException` would cause a `SEGFAULT`, and that is not ideal. So each function exposed to Rust that can throw has a `try ... catch`. If an exception is caught the stack unwinding happens in the C++ library, which knows how it handle it (hopefully), and it returns a `sgobject_result` type, which has a `union`. The `union` either holds a `sgobject_t` pointer or a `const char*`, which is the error message. Unions are part of Rust so the rest of the implementation is pretty straighforward. And we can even use my favourite Rust feature: [pattern matching](https://doc.rust-lang.org/book/ch18-03-pattern-syntax.html)! Do not worry about the complicated type names, or `#name`, the main point is that we can access a C `union` through `bindgen` and bypass exceptions.
```rust
match result {
	sgobject_result { return_code: sgobject_result_SUCCESS,
	                  result: sgobject_result_ResultUnion { result: ptr } } => Ok(#name { ptr }),
	sgobject_result { return_code: sgobject_result_ERROR,
	                  result: sgobject_result_ResultUnion { error: msg } } => {
	                  	let c_error_str = CStr::from_ptr(msg);
	                  	Err(format!("{}", c_error_str.to_str().expect("Failed to get error")))
	},
	_ => Err(format!("Unexpected return.")) // this should be impossible to reach
```

# Conclusion
I wanted to cover in this blog post how to train a Shogun machine learning algorithm from Rust. However, this post got a bit long, and quite heavy in the details. So I will be covering this part in a future post!

Overall, I found calling C++ from Rust fairly straightforward, but I did use C as a bridge, which made my life much easier. From the past week I have been working on this I learnt that further C++ support might be coming soon (for example [Rust nightly now supports C++ stack unwinding](https://github.com/rust-lang/rust/pull/65646)), but for a simple library interface I quickly realised that I was not willing to go down that rabbit hole. Maybe one day I will try to rewrite the wrappers to work straigth out of the box with Shogun's C++ code, but for now I am happy that [`shogun-rust`](https://github.com/gf712/shogun-rust) is starting to take shape!