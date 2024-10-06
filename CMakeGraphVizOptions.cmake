# To generate a dependency graph of the entire CMake project, pass the
# `--graphviz=build/project.dot` flag during configuration, and then use
# `dot -Tpdf -O build/project.dot` to generate a PDF of the graph.

set(GRAPHVIZ_GENERATE_DEPENDERS Off)
set(GRAPHVIZ_IGNORE_TARGETS
    "^warnings$"
    "(^|::)common_options$"
    "^benchmark-"
    "^Threads::Threads$"
    "^Doxygen::doxygen$"
    "^Python3?::Interpreter$"
    "^(-l)?dl$"
    "^(-l)?gomp$"
    "^(-l)?m$"
    "^(-l)?pthread$"
    "^(-l)?rt$"
    "^MKL::mkl_"
    "^CONAN_LIB::"
    "_DEPS_TARGET$"
    "/libpthread\\.so"
    "/libgomp\\.so"
    "/MATLAB/"
    "-Wl,--start-group"
    "-Wl,--end-group"
)
