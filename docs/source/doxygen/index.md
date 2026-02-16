Fast linear algebra routines for batches of small matrices.

Batmat is used as the linear algebra backend for the [Cyqlone](https://github.com/kul-optec/cyqlone) solver,
where it is used to perform vectorized operations across multiple stages in an optimal control problem.

To enable vectorization, batmat stores batches of small matrices in an interleaved “compact” format in memory, where the corresponding elements of all matrices in a batch are stored together.
Custom linear algebra routines then operate on all matrices in a batch simultaneously using SIMD instructions. These routines are built on top of highly optimized micro-kernels.

For an overview of the supported routines, see @ref topic-linalg. For documentation of the batched
matrix data structures, see @ref topic-matrix.

---

<div style="text-align: center;">
    <img src="batmat-small.png" alt="batmat logo" width=160>
</div>
