import os

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.build import can_run
from conan.tools.files import save
from conan.tools.scm import Git


class BatmatRecipe(ConanFile):
    name = "batmat"
    version = "0.0.12"

    license = "LGPL-3.0-or-later"
    author = "Pieter P <pieter.p.dev@outlook.com>"
    url = "https://github.com/tttapa/batmat"
    description = "Fast linear algebra routines for batches of small matrices."
    topics = "scientific software"

    # Binary configuration
    package_type = "library"
    settings = "os", "compiler", "build_type", "arch"
    bool_batmat_options = {
        "with_openmp": False,
        "with_benchmarks": False,
        "with_cpu_time": False,
        "with_gsi_hpc_simd": False,
        "with_blasfeo": False,
    }
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "vector_lengths_double": ["ANY"],
        "vector_lengths_float": ["ANY"],
        "dtypes": ["double", "double,float", "ANY"],
    } | {k: [True, False] for k in bool_batmat_options}
    default_options = {
        "shared": False,
        "fPIC": True,
        "dtypes": "double",
    } | bool_batmat_options

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = (
        "CMakeLists.txt",
        "src/*",
        "cmake/*",
        "interfaces/*",
        "test/*",
        "benchmarks/*",
        "LICENSE",
        "README.md",
    )

    def export_sources(self):
        git = Git(self)
        status_cmd = "status . --short --no-branch --untracked-files=no"
        dirty = bool(git.run(status_cmd).strip())
        hash = git.get_commit() + ("-dirty" if dirty else "")
        print("Commit hash:", hash)
        save(self, os.path.join(self.export_sources_folder, "commit.txt"), hash)

    generators = ("CMakeDeps",)

    def requirements(self):
        self.requires("guanaqo/1.0.0-alpha.22", transitive_headers=True, transitive_libs=True)
        if self.options.get_safe("with_benchmarks"):
            self.requires("benchmark/1.9.4")
            self.requires("hyhound/1.1.1")
        if self.options.get_safe("with_openmp") and self.settings.compiler == "clang":
            self.requires(f"llvm-openmp/[~{self.settings.compiler.version}]")
        if self.options.get_safe("with_gsi_hpc_simd"):
            self.requires("gsi-hpc-simd/tttapa.20250625", transitive_headers=True)
        if self.options.get_safe("with_blasfeo"):
            self.requires("blasfeo/tttapa.20260119")

    def build_requirements(self):
        self.test_requires("eigen/[~5.0]")
        self.test_requires("gtest/1.17.0")
        self.tool_requires("cmake/[>=3.24 <5]")

    def config_options(self):
        if self.settings.get_safe("os") == "Windows":
            self.options.rm_safe("fPIC")
        # Set architecture-dependent defaults for vector lengths
        arch = str(self.settings.arch)
        if arch in ["armv7hf", "armv8", "armv8_32", "armv8.3", "arm64ec"]:
            # ARM NEON: 128-bit vectors (2x double)
            self.options.vector_lengths_double = "1,2,4"
            self.options.vector_lengths_float = "1,4,8"
        elif arch in ["x86_64"]:
            # x86_64 AVX2/AVX-512: 256-bit and 512-bit vectors (4x and 8x double)
            self.options.vector_lengths_double = "1,4,8"
            self.options.vector_lengths_float = "1,4,8,16"
        else:
            # Conservative defaults for other architectures
            self.options.vector_lengths_double = "1,2,4"
            self.options.vector_lengths_float = "1,4,8"

    def configure(self):
        if self.options.get_safe("with_benchmarks"):
            self.options["guanaqo/*"].with_blas = True

    def layout(self):
        cmake_layout(self)
        self.cpp.build.builddirs.append("")

    def generate(self):
        tc = CMakeToolchain(self)
        for k in self.bool_batmat_options:
            value = self.options.get_safe(k, None)
            if value is not None and value.value is not None:
                tc.variables["BATMAT_" + k.upper()] = bool(value)
        dtypes = str(self.options.dtypes).replace(",", ";")
        tc.variables["BATMAT_DTYPES"] = dtypes
        dtypes_list = [dt.strip() for dt in dtypes.split(";")]
        if "double" in dtypes_list:
            vl_double = str(self.options.vector_lengths_double).replace(",", ";")
            tc.variables["BATMAT_VECTOR_LENGTHS_DOUBLE"] = vl_double
        if "float" in dtypes_list:
            vl_float = str(self.options.vector_lengths_float).replace(",", ";")
            tc.variables["BATMAT_VECTOR_LENGTHS_FLOAT"] = vl_float
        guanaqo = self.dependencies["guanaqo"]
        index_type = guanaqo.options.get_safe("blas_index_type", default="int")
        tc.variables["BATMAT_DENSE_INDEX_TYPE"] = index_type
        if can_run(self):
            tc.variables["BATMAT_FORCE_TEST_DISCOVERY"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        cmake.test()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs.append(os.path.join("lib", "cmake", "batmat"))
