import os

from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout
from conan.tools.build import can_run
from conan.tools.files import save
from conan.tools.scm import Git


class BatmatRecipe(ConanFile):
    name = "batmat"
    version = "1.0.0"

    # Optional metadata
    license = "LGPLv3"
    author = "Pieter P <pieter.p.dev@outlook.com>"
    url = "https://github.com/tttapa/batmat"
    description = "Linear solvers and adapters for solving KKT systems."
    topics = "scientific software"

    # Binary configuration
    package_type = "library"
    settings = "os", "compiler", "build_type", "arch"
    bool_batmat_options = {
        "with_openblas": True,
        "with_mkl": False,
        "with_openmp": False,
        "with_benchmarks": False,
        "with_python": False,
        "with_cpu_time": False,
        "with_gsi_hpc_simd": False,
    }
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "dense_index_type": ["int", "long", "long long"],
    } | {k: [True, False] for k in bool_batmat_options}
    default_options = {
        "shared": False,
        "fPIC": True,
        "dense_index_type": "long long",
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
        self.requires(
            "guanaqo/1.0.0-alpha.15", transitive_headers=True, transitive_libs=True, force=True
        )
        if self.options.with_openblas:
            self.requires("openblas/0.3.27", transitive_headers=True)
        if self.options.with_benchmarks:
            self.requires("benchmark/1.8.4")
        if self.options.with_python:
            self.requires("eigen/tttapa.20240516", force=True)
        else:
            self.test_requires("eigen/tttapa.20240516", force=True)
        self.test_requires("gtest/1.15.0")
        if self.options.with_python:
            self.requires("pybind11/2.13.6")
        if self.options.with_openmp and self.settings.compiler == "clang":
            self.requires(f"llvm-openmp/[~{self.settings.compiler.version}]")
        if self.options.with_gsi_hpc_simd:
            self.requires("gsi-hpc-simd/tttapa.20250625")

    def config_options(self):
        if self.settings.get_safe("os") == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        # There is currently no 64-bit indices option for OpenBLAS using Conan
        if self.options.with_openblas:
            self.options.rm_safe("dense_index_type")
        self.options["guanaqo/*"].with_blas = True

    def layout(self):
        cmake_layout(self)
        self.cpp.build.builddirs.append("")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BATMAT_DENSE_INDEX_TYPE"] = self.options.get_safe(
            "dense_index_type", default="int"
        )
        print(tc.variables["BATMAT_DENSE_INDEX_TYPE"])
        for k in self.bool_batmat_options:
            value = getattr(self.options, k, None)
            if value is not None and value.value is not None:
                tc.variables["BATMAT_" + k.upper()] = bool(value)
        if self.options.with_python:
            tc.variables["USE_GLOBAL_PYBIND11"] = True
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
