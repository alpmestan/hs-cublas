This is an ongoing work on a binding to the CUBLAS library. This will eventually be used in my upcoming `accelerate-cublas` library that will expose CUBLAS's features through operations on `Accelerate` arrays.

Installation
============

To check that it works for you, make sure you install the CUDA toolkit, which includes the CUBLAS library. You can see the instructions [here](http://hackage.haskell.org/package/accelerate-cuda).

Once it's done, from the root dir:

```
$ cabal sandbox init
$ cabal install
```

This will use the `configure` script to make sure you have everything installed, and build the library and the `cublas-test` executable. For now, it just asks the CUBLAS library for its version and prints it, using the initialization and cleanup routines from CUBLAS.

Later, this will be replaced by an actual test suite. For now, running it is as simple as:

```
$ .cabal-sandbox/bin/cublas-test
CUBLAS version: 5050
```

