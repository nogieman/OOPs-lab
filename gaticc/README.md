# gaticc - Compiler/Simulator/Runtime for Gati DNN accelerator

# Build

## Install Dependencies 

**Arch**:
```
sudo pacman -S python3 python-pip pkg-config cmake
```

**Fedora**:
``` 
sudo dnf install python3-devel python-pip cmake
```

**Ubuntu/Debian**:
```
sudo apt install python3-dev python3 python3-pip pkg-config cmake
```

**MacOs**
```
brew install python pkg-config cmake
```

### On board

On the board, you would need additional dependencies that allow CPU-FPGA communication:

First, check if "Vaaman FPGA communication" is checked in the overlay config,
find a detailed how-to
[here](https://docs.vicharak.in/vaaman-linux/linux-configuration-guide/vicharak-config-tool/#vicharak-config-overlays).
Consider, rebooting after this.

Next, add Vicharak's apt repo to your board's apt config. Follow [this
guide](https://docs.vicharak.in/vicharak_sbcs/vaaman/vaaman-linux/linux-configuration-guide/vicharak-apt-config/)
to do this.

After this is setup, run:

```
sudo apt update && sudo apt upgrade
sudo apt install rah-service
```

## Compile

```
cd /path/to/gaticc
./scripts/install_deps.sh
mkdir build
cmake -B build
cmake --build build && sudo cmake --install build
sudo pip install -e .
```

# Usage

See,
```
gaticc -h
```
for usage instructions.

## Python Interface

Here's an example script to run simulation of a model (install model files from the model zoo):
```
import gati

onnx_path = "tests/models/mnist_6_28_int8.onnx"
print(np.argmax(np.squeeze(gati.sim(onnx_path, np.load("mnist_2.py")), axis=1), axis=-1))
```

For more examples, checkout `examples/` directory. It contains, scripts to
compile, run, summarize etc.  

For full api doc, read `python/gati.py` (or feed it to your favorite LLM).

# Versioning

Gaticc uses three numbers in the style of <https://semver.org/>:

```
MAJOR.MINOR.PATCH
```

The version number is:

1. Used to track the history of this program
2. **Keep it in sync with the hardware** (i.e. the Gati project)

This is done by assigining a meaning to each number and agreeing with the
hardware maintainers on when to increment which number. 

To keep in sync, the major and minor numbers should always be equal to
that of the hardware. So, if we ask ourselves, which version of the 
hardware is compatible with gaticc version 1.3.x, the answer is:
Gati version 1.3.x. Keeping the compatibility intact is in the hands
of the maintainers of both projects. The patch number in both cases
should be the latest available for that `major.minor` combination.

When are version numbers incremented:

- major: when architectures are changed fundamentally. for example, a move
from 9x4x4 SAs to 9x8x8, or 9x8x8 to Mobilenet. 
- minor: when a change is supposed to take place in both hardware and software.
for example, addition of an extra field in some instruction. this requires both
hardware and software to be changed to implemented this feature leading to a 
minor version bump.
- patch: patch numbers are incremented changes agnostic to hardware are made
in the software. for example, when a segfault is patched in the software. as
this demands no change in the hardware, only the patch numbers are increased.

Ideally, a version `a.b.c` is always compatible with the version
`(a-1).(b-1).(c-1)`.

Version bumps are controlled manually by the maintainer (who can push to master)
through the `bump_version.sh` script present in the root of this repo. Read the
script (or `./bump_version.sh` to get usage message) to understand what all it
does. Any push to master should be preceded by a version bump.

# Contribution Guidelines

- Format all your commit messages according to <https://www.conventionalcommits.org/en/v1.0.0/>.
- For bugs, create an issues here: <https://github.com/vicharak-in/gaticc/issues/>
- Keep commit message titles succinct. Use the body for further elaboration if
  needed. See <https://cbea.ms/git-commit/>
- PRs should be related to a topic/goal, be easy to review and check for bugs.
  Do not create large PRs with random changes. 
- Write simple and easy to read/maintain code.
- "Pre-mature optimizations are the root of all evil". Measure before you
  optimize. Do not use convoluted features of a language just because you know
  them.
- Read <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines> and what
  and what not to use.
- Use a formatter. gaticc uses LLVM style formatting for c++
  <https://clang.llvm.org/docs/ClangFormat.html>
- We do not follow any coding guideline to the T but welcome good style
  suggestions. (suggest through ISSUES)
- You can find some here: <https://google.github.io/styleguide/cppguide.html>

# Tests

In the `tests/` directory individual python scripts are used to 
test primary function of gaticc. These files are:

- `test_compile.py`
- `test_sim.py`
- `test_summary.py`
- `test_dispatch.py`

See their `-h` help messages to understand how they ought to be used. 

# Model Zoo

[Download models and other files](http://galactos.local:8471/)
