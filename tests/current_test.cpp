// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "common.hpp"
#include "common/bfloat16.hpp"
#include "stream.hpp"

TEST(CurrentTests, B16EltwiseSAXPY) {
    uint32_t count = 1024 * 4;
    auto type = tt::DataFormat::Float16_b;
    uint32_t n_tiles = std::ceil(count / TILE_SIZE);

    int seed = std::random_device{}();
    std::mt19937 rng(seed);

    // Stream data.
    constexpr auto max_float = 10.0F;
    // std::vector<uint32_t> generator0_data = create_random_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type),
    // max_float, seed); std::vector<uint32_t> generator1_data = create_random_vector_of_bfloat16(TILE_SIZE * n_tiles *
    // datum_size(type), max_float, seed);
    std::vector<uint32_t> generator0_data =
        create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), -1.0F);
    std::vector<uint32_t> generator1_data =
        create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), 4.0F);
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
    auto max_parallelization_factor = 4;
    current::Map map({&kernel_a}, {&source0, &source1, &sink}, max_parallelization_factor);
    map.add_connection(&source0, &kernel_a, "in0");
    map.add_connection(&source1, &kernel_a, "in1");
    map.add_connection(&kernel_a, "out0", &sink);

    // Execute program.
    map.execute();

    // Validate output.
    auto out = map.read_stream(&sink);
    auto out_bf16 = unpack_uint32_vec_into_bfloat16_vec(out);
    EXPECT_EQ(out_bf16.size(), TILE_SIZE * n_tiles);
    for (size_t i = 0; i < out_bf16.size(); i++) {
        std::cout << i << ": " << out_bf16[i].to_float() << "\n";
    }
    std::cout << "\n";

    auto in0 = unpack_uint32_vec_into_bfloat16_vec(generator0_data);
    auto in1 = unpack_uint32_vec_into_bfloat16_vec(generator1_data);
    bool pass = true;
    for (size_t i = 0; i < out_bf16.size(); i++) {
        // Check that out[i] = in0[i] * 2.0 + in1[i]
        auto expected = bfloat16(in0[i].to_float() * 2.0F + in1[i].to_float());
        // auto expected = bfloat16(2.0F);
        pass &= is_close(expected.to_float(), out_bf16[i].to_float());
    }
    EXPECT_TRUE(pass);
}