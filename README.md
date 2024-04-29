# cubemap_converter

Tool for remapping oversampled cube-maps into single images by applying camera intrinsics.

## Build steps

Dependencies:
- cmake
- libpng
- zlib
- OpenGL
- On windows, I suggest using Ninja as the build system.

Clone and fetch submodules:
```bash
git clone git@github.com:gareth-cross/cubemap_converter.git
cd cubemap_converter
git submodule update --init --recursive
```

Configure and build:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -Wno-deprecated -Wno-dev -G Ninja
cmake --build .
```

## Running:

The suggested way to execute the tool is via the `convert_data.py` script:

```bash
python scripts/convert_data.py --config <PATH TO TOML> --input <INPUT DATASET DIRECTORY> --output <OUTPUT DATASET DIRECTORY>
```

Where:
- `<INPUT DATASET DIRECTORY>` is the directory containing the output of Unreal Engine.
- `<OUTPUT DATASET DIRECTORY>` is the directory in which the converted PNGs will be saved.

This command will:
1. Load the calibrations from the TOML file.
2. For each camera, generate a remap table from the intrinsics.
3. Execute the `cubemap_converter` with the given intrinsic model.

The number of cameras in the dataset directory should match the number of cameras in the TOML file. See the [scripts](/scripts) directory for example configurations.
