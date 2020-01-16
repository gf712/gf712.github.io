---
layout: post
title:  "Improving C++ performance with SIMD: a Bioinformatics case study"
categories: [programming]
tags: [C++, optimisation, bioinformatics]
---

## Introduction
I was recently working on optimising some C++ code that does antibody numbering (if you want to find out more about the topic checkout the web server for the original C code [here](http://bioinf.org.uk/abs/abnum/)).
I did the usual analysis with linux's `perf` to find some hotspots. These included a call to `std::log2` that I replaced using a low precision version that I found [here](https://www.flipcode.com/archives/Fast_log_Function.shtml), and an argsort that was rewritten with a heap data structure (more on this in a future post). After all this work, the executable spent around 27-28% of the time with calls to [`std::string::find`](https://en.cppreference.com/w/cpp/string/basic_string/find).

## The function
The function in question is actually very simple, but the code is surprisingly innefficient. This function, let's call it `get_score_for_residue`, takes an amino acid, a list of scores for various residues and another list with the residue names. It tries to find the amino acid in the list of residues and uses the position to get the score from the score list. This is how it looks like:
```cpp
double get_score_for_residue(
	char residue, const std::vector<double>& scores, const std::string& residues)
{
	auto pos = residues.find(residue);
	if (pos != std::string::npos)
		return scores[pos];
	else
		return 0.05;
}
```
The else condition returns 0.05, which is the average score of a residue (we expect there to be 20 amino acids, and the sum of the scores is 1). This happens when `residue` is not found in `residues`. It turns out that almost 30% of the program execution goes into `auto pos = residues.find(residue);`, because `get_score_for_residue` is called in a loop.

## Optimising code with SIMD
I personally thought that this code was properly vectorised when using `-O3` and `-march=native	` on my machine which has SSE4.2 and AVX512 instruction available. However, the compiler didn't do this, or at least not to the extent I was expecting. So I tried to do it myself, and it actually improved my runtime significantly! By the way, this code is based on code shown in a presentation about abseils' hashtable implementation which you can checkout [here](https://www.youtube.com/watch?v=JZE3_0qvrMg). This is all written for SSE2, so most people should be able to compile this if they have an Intel CPU. If you don't know what SSE is, have a look at wikipedia's [page](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) on that topic. In a nutshell, a single instruction is executed on all values in a CPU register, which means that on a 128-bit register you can execute a single instruction on 4 32-bit integers in one go, rather than doing it four times (one instruction for each value).

## Optimising `find` with SSE2
In this case I want to optimise a find operation with `char`, so I can actually do a lookup of 16 residues in one go (assuming `char` is 8 bits, 128-bit register / 8 = 16). So, we need to load 16 char from `residues` into a register, then `residue` into another register (which will have the same value 16 times), and then compare `residues` with `residue`. This will give me a 32-bit integer where one bit set is to 1 where `residue` is equal to a `char` in `residues` (note that only the lower half is of interest, i.e. the 16 least significant bits). We assume there is only one such residue, and get the position of the match. Also you may have noticed that 16 bytes is not enough for the expected size of 20 for residues (there are 20 natural amino acids), so we need to do this operation twice, if a match was not found in the first 16 bytes. The bit position is found with a trick I found in this stackoverflow [answer](https://stackoverflow.com/a/757266), or if you use clang or gcc you can use [`__builtin_ffs`](https://en.wikipedia.org/wiki/Find_first_set).
Alright, this should be enough theory, so now some code!
```cpp
double get_score_for_residue(
	char residue, const std::vector<double>& scores, const std::string& residues)
{
	size_t pos = std::string::npos;
	size_t offset = 0;
	constexpr size_t register_width = 128 / 8;

	// load first 16 char of string
	const __m128i src = _mm_load_si128((__m128i*)residues.c_str());
	// set the register with 16 char equals to the query residue
	const __m128i query_128 = _mm_set1_epi8(residue);
	// do a bitwise comparisson of all the bits between the two registers
	const __m128i mask = _mm_cmpeq_epi8(query_128, src);
	// get the mask as a C++ int type
	int result = _mm_movemask_epi8(mask);
	// if the result was not found in the first 16 char in the string
	// do the same for the second part
	if (result == 0)
	{
		// same instructions as above but with a 16 byte offset
		const __m128i src = _mm_load_si128((__m128i*)(residues.c_str() + register_width));
		const __m128i mask = _mm_cmpeq_epi8(query_128, src);
		result = _mm_movemask_epi8(mask);
		offset += register_width;
	}

	// if the result was not found in the first 32 char (2*16), then pos=std::string::npos
	// otherwise we set it to the position in which a match was found
	if (result != 0)
		pos = offset + bit_pos(result);

	if (pos != std::string::npos)
		return scores[pos];
	else
		return 0.05;
}
```

Obviously this could be written in a loop, rather than copy paste the instructions inside the `if (result == 0)`. But in my case it is safe to assume that `std::vector<double> residues` will never be larger than 32. After rerunning `perf`, I concluded that now only 7.5% of the execution of the program is dedicated to finding `residue` in `residues`, and the total runtime is decreased!

## Conclusion
Just to give you some figures, without the SSE2 version I was running the program for 1 minute, and with the faster version, this went down to 24 seconds! So, `std::string::find` runs for about 16.8 seconds (60 seconds * 28% = 16.8 seconds) originally and the SSE version only took 1.8 seconds (24 seconds * 7.5% = 1.8 seconds), which is an almost ten fold speed up!

## PS
For those who are interested in optimising code and use a linux distribution I recommend using [`hotspot`](https://github.com/KDAB/hotspot), which provides an amazing GUI for `perf` results!
