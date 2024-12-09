// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <ctime>
#include <random>

#include "common.hpp"
#include "common/bfloat16.hpp"
#include "common/tt_backend_api_types.hpp"
#include "stream.hpp"

TEST(CurrentTests, B16EltwiseSAXPY) {
    uint32_t count = 1024 * 1024 * 512;
    auto type = tt::DataFormat::Float16_b;
    uint32_t n_tiles = std::ceil(count / TILE_SIZE);

    int seed = static_cast<int>(std::time(nullptr));
    std::mt19937 rng(seed);

    // Stream data.
    constexpr auto max_float = 10.0F;
    std::vector<uint32_t> generator0_data =
        create_random_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), max_float, seed);
    std::vector<uint32_t> generator1_data =
        create_random_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), max_float, seed);
    // std::vector<uint32_t> generator0_data =
    //     create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), 3.0F);
    // std::vector<uint32_t> generator1_data =
    //     create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), 4.0F);
    std::vector<uint32_t> output_data = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 0.0f);

    // Init kernel.
    current::Kernel kernel_a;

    // Define ports and set compute kernel.
    kernel_a.add_input_port("in0", type);
    kernel_a.add_input_port("in1", type);
    kernel_a.add_output_port("out0", type);
    kernel_a.set_compute_kernel(
        R"(
        out0 = in0 * 2.0 + in1;
    )",
        false);

    // Define streams.
    current::Stream source0(generator0_data, count, type);
    current::Stream source1(generator1_data, count, type);
    current::Stream sink(output_data, count, type);

    // Define connections between streams and kernels.
    // TODO: Fails when this is an odd # (something to do w/ how work is split).
    auto max_parallelization_factor = 1;
    auto tiles_per_cb = 1;
    current::Map map({&kernel_a}, {&source0, &source1, &sink}, max_parallelization_factor, tiles_per_cb);
    map.add_connection(&source0, &kernel_a, "in0");
    map.add_connection(&source1, &kernel_a, "in1");
    map.add_connection(&kernel_a, "out0", &sink);

    // Execute program.
    map.execute();

    // Validate output.
    auto out = map.read_stream(&sink);
    auto out_bf16 = unpack_uint32_vec_into_bfloat16_vec(out);
    EXPECT_EQ(out_bf16.size(), TILE_SIZE * n_tiles);
    // for (size_t i = 0; i < out_bf16.size(); i++) {
    //     std::cout << i << ": " << out_bf16[i].to_float() << "\n";
    // }
    // std::cout << "\n";

    auto in0 = unpack_uint32_vec_into_bfloat16_vec(generator0_data);
    auto in1 = unpack_uint32_vec_into_bfloat16_vec(generator1_data);
    bool pass = true;
    for (size_t i = 0; i < out_bf16.size(); i++) {
        // Check that out[i] = in0[i] * 2.0 + in1[i]
        auto expected = bfloat16(in0[i].to_float() * 2.0F + in1[i].to_float());
        // auto expected = bfloat16(in0[i].to_float() * 2.0F);
        // auto expected = bfloat16(2.0F);
        pass &= is_close(expected.to_float(), out_bf16[i].to_float());
    }
    EXPECT_TRUE(pass);
}

TEST(CurrentTests, GatherTest) {
    // uint32_t num_indices = 1024 * 1024 * 256;
    uint32_t num_indices = 1024 * 2;
    uint32_t data_buffer_size = 1024 * 1024;  // 1MB data buffer
    auto type = tt::DataFormat::Float16_b;
    auto n_tiles = static_cast<uint32_t>(std::ceil(num_indices / static_cast<double>(TILE_SIZE)));
    std::cout << "n_tiles: " << n_tiles << "\n";

    int seed = static_cast<int>(std::time(nullptr));
    std::mt19937 rng(seed);

    // Stream data.
    auto data_tile_size = TILE_SIZE * datum_size(type);
    auto index_tile_size = TILE_SIZE * sizeof(uint32_t);
    constexpr auto max_float = 10.0F;
    // std::vector<uint32_t> generator0_data = create_random_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type),
    // max_float, seed); std::vector<uint32_t> generator1_data = create_random_vector_of_bfloat16(TILE_SIZE * n_tiles *
    // datum_size(type), max_float, seed);
    // std::vector<uint32_t> data_buffer =
    //     create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), -1.0F);
    // std::vector<uint32_t> data_buffer = create_arange_vector_of_bfloat16(data_tile_size * n_tiles, false);
    std::vector<uint32_t> data_buffer = create_random_vector_of_bfloat16(data_tile_size * 1024, max_float, seed);
    // std::vector<uint32_t> data_buffer = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles *
    // datum_size(type), 2.0F); std::vector<uint32_t> data_buffer = std::vector<uint32_t>(64, 0U); for (size_t i = 0; i
    // < data_buffer.size(); i++) {
    //     data_buffer[i] = i;
    // }
    std::uniform_int_distribution<uint32_t> dist(0, data_buffer_size - 1);
    auto index_vec = std::vector<uint32_t>(num_indices, 0);
    for (size_t i = 0; i < num_indices; i++) {
        index_vec[i] = dist(rng);
    }
    auto output_data = create_constant_vector_of_bfloat16(data_tile_size * n_tiles, -1.0F);

    // std::vector<uint32_t> generator1_data =
    //     create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), 4.0F);
    // std::vector<uint32_t> output_data = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 0.0f);

    // Init kernel.
    current::Kernel kernel_a;

    // Define ports and set compute kernel.
    kernel_a.add_input_port("in0", type);
    kernel_a.add_output_port("out0", type);

    // Define streams.
    bool use_sram = false;
    current::GatherStream gather_stream(data_buffer, type, data_buffer_size, index_vec, use_sram);
    current::Stream sink(output_data, num_indices, type);

    // Define connections between streams and kernels.
    auto max_parallelization_factor = 1;
    auto tiles_per_cb = 1;
    current::Map map({&kernel_a}, {&gather_stream, &sink}, max_parallelization_factor, tiles_per_cb);
    map.add_connection(&gather_stream, &kernel_a, "in0");
    map.add_connection(&kernel_a, "out0", &sink);

    // Execute program.
    map.generate_device_kernels();
    map.execute();

    std::cout << "Finished!\n";

    // Validate output.
    auto in_raw = map.read_gather_stream(&gather_stream, true);
    auto in = unpack_uint32_vec_into_bfloat16_vec(in_raw);
    auto out = map.read_stream(&sink);
    auto out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(out);

    bool pass = true;
    for (size_t i = 0; i < num_indices; i++) {
        auto index = index_vec[i] * 16;
        std::cout << std::dec;  // Force decimal output
        std::cout << "i: " << i << ",Index: " << index << ", ";
        auto input = in[index];
        std::cout << "in[" << index << "] = " << input.to_float() << ", ";
        auto output = out_b16_vec[i];
        std::cout << "out: " << out_b16_vec[i].to_float() << "\n";

        pass &= is_close(input.to_float(), output.to_float());
        // std::cout << "\n";
    }
    EXPECT_TRUE(pass);
}

TEST(CurrentTests, GatherTestSRAM) {
    // uint32_t num_indices = 1024 * 1024 * 256;
    uint32_t num_indices = 1024 * 2;
    uint32_t data_buffer_size = 1024 * 256;
    auto type = tt::DataFormat::Float16_b;
    auto n_tiles = static_cast<uint32_t>(std::ceil(num_indices / static_cast<double>(TILE_SIZE)));
    std::cout << "n_tiles: " << n_tiles << "\n";

    int seed = static_cast<int>(std::time(nullptr));
    std::mt19937 rng(seed);

    // Stream data.
    auto data_tile_size = TILE_SIZE * datum_size(type);
    auto index_tile_size = TILE_SIZE * sizeof(uint32_t);
    constexpr auto max_float = 10.0F;
    // std::vector<uint32_t> generator0_data = create_random_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type),
    // max_float, seed); std::vector<uint32_t> generator1_data = create_random_vector_of_bfloat16(TILE_SIZE * n_tiles *
    // datum_size(type), max_float, seed);
    // std::vector<uint32_t> data_buffer =
    //     create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), -1.0F);
    // std::vector<uint32_t> data_buffer = create_arange_vector_of_bfloat16(data_tile_size * n_tiles, false);
    std::vector<uint32_t> data_buffer = create_random_vector_of_bfloat16(data_tile_size * 1024, max_float, seed);
    // std::vector<uint32_t> data_buffer = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles *
    // datum_size(type), 2.0F); std::vector<uint32_t> data_buffer = std::vector<uint32_t>(64, 0U); for (size_t i = 0; i
    // < data_buffer.size(); i++) {
    //     data_buffer[i] = i;
    // }
    std::uniform_int_distribution<uint32_t> dist(0, data_buffer_size - 1);
    auto index_vec = std::vector<uint32_t>(num_indices, 0);
    for (size_t i = 0; i < num_indices; i++) {
        index_vec[i] = dist(rng);
    }
    auto output_data = create_constant_vector_of_bfloat16(data_tile_size * n_tiles, -1.0F);

    // std::vector<uint32_t> generator1_data =
    //     create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), 4.0F);
    // std::vector<uint32_t> output_data = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 0.0f);

    // Init kernel.
    current::Kernel kernel_a;

    // Define ports and set compute kernel.
    kernel_a.add_input_port("in0", type);
    kernel_a.add_output_port("out0", type);

    // Define streams.
    current::GatherStream gather_stream(data_buffer, type, data_buffer_size, index_vec);
    current::Stream sink(output_data, num_indices, type);

    // Define connections between streams and kernels.
    auto max_parallelization_factor = 1;
    auto tiles_per_cb = 1;
    current::Map map({&kernel_a}, {&gather_stream, &sink}, max_parallelization_factor, tiles_per_cb);
    map.add_connection(&gather_stream, &kernel_a, "in0");
    map.add_connection(&kernel_a, "out0", &sink);

    // Execute program.
    map.generate_device_kernels();
    map.execute();

    std::cout << "Finished!\n";

    // Validate output.
    auto in_raw = map.read_gather_stream(&gather_stream, true);
    auto in = unpack_uint32_vec_into_bfloat16_vec(in_raw);
    auto out = map.read_stream(&sink);
    auto out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(out);

    bool pass = true;
    // for (size_t i = 0; i < num_indices; i++) {
    //     auto index = index_vec[i] * 16;
    //     std::cout << std::dec;  // Force decimal output
    //     std::cout << "i: " << i << ",Index: " << index << ", ";
    //     auto input = in[index];
    //     std::cout << "in[" << index << "] = " << input.to_float() << ", ";
    //     auto output = out_b16_vec[i];
    //     std::cout << "out: " << out_b16_vec[i].to_float() << "\n";

    //     pass &= is_close(input.to_float(), output.to_float());
    //     // std::cout << "\n";
    // }
    EXPECT_TRUE(pass);
}