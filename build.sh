#!/bin/bash

# Default build type
BUILD_TYPE="Release"
CLEAN=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Display help
if [ "$HELP" = true ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d, --debug    Build in Debug mode"
    echo "  -r, --release  Build in Release mode (default)"
    echo "  -c, --clean    Clean build directory before building"
    echo "  -h, --help     Show this help message"
    exit 0
fi

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory if it doesn't exist
mkdir -p build

# Go to build directory
cd build

# Run cmake if CMakeCache.txt doesn't exist (first time) or if CMakeLists.txt was modified
if [ ! -f CMakeCache.txt ] || [ ../CMakeLists.txt -nt CMakeCache.txt ] || [ "$CLEAN" = true ]; then
    echo "Configuring CMake with BUILD_TYPE=$BUILD_TYPE..."
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
fi

# Build the project
echo "Building..."
make -j$(nproc)  # Uses all available CPU cores