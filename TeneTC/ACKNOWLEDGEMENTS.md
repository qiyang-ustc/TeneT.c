# Acknowledgements

TeneT.c builds on the ideas and interface conventions of TeneT.jl. We thank
Xingyu Zhang and the TeneT.jl contributors for the original package.

Release comparisons use TeneT.jl as a reference implementation at a pinned
commit. When modern CUDA or KrylovKit package versions need a small compatibility
adapter to run that reference baseline, the adapter is recorded in the benchmark
artifact and should not be interpreted as a defect in the original work.

TeneT.c also depends on KrylovKit.c, which in turn uses KrylovKit.jl as its
semantic reference. We thank the KrylovKit.jl authors and contributors as well.

