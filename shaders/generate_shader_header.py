"""
Script that turns GLSL file into C++ header so we can include it directly.

Sad that this is necessary, but maybe eventually we'll get something like `include_bytes!`
"""
import argparse
import os


def main(args: argparse.Namespace):
    filename = os.path.basename(args.input)
    shader_name, _ = os.path.splitext(filename)

    with open(args.input) as handle:
        contents = handle.read()

    output = str()
    output += '// Machine generated file - do not modify.\n'
    output += f'// Generated from: {filename}\n'
    output += '#pragma once\n#include <string_view>\n\n'
    output += 'namespace shaders {\n'
    output += f'constexpr std::string_view {shader_name} = R"(\n'
    output += contents
    output += '\n)";\n'
    output += '}  // namespace shaders\n'

    with open(args.output, 'w') as handle:
        handle.write(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=str, required=True, help="Input shader file")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
