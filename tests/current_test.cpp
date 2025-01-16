// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <ctime>
#include <random>

#include "common.hpp"
#include "common/bfloat16.hpp"
#include "common/tt_backend_api_types.hpp"
#include "map.hpp"
#include "stream.hpp"

TEST(CurrentTests, Pipeline) {
    // TODO: Fails when num tiles > 2^16?
    uint32_t count = 1024 * 1024 * 64;
    auto type = tt::DataFormat::Float16_b;
    uint32_t n_tiles = std::ceil(count / TILE_SIZE);

    int seed = static_cast<int>(std::time(nullptr));
    std::mt19937 rng(seed);

    // Stream data.
    constexpr auto max_float = 10.0F;
    std::vector<uint32_t> generator0_data =
        create_random_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), max_float, seed);
    std::vector<uint32_t> output_data =
        create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), -1.0F);

    // Init kernel.
    current::Kernel kernel_a;
    current::Kernel kernel_b;

    // Define ports and set compute kernel.
    kernel_a.add_input_port("in0", type);
    kernel_a.add_output_port("out0", type);
    kernel_b.add_input_port("in0", type);
    kernel_b.add_output_port("out0", type);
    // kernel_a.set_compute_kernel(
    //     R"(
    //     out0 = in0 * 2.0 + in1;
    // )",
    //     false);

    // Define streams.
    current::Stream source0(generator0_data, count, type);
    current::Stream sink(output_data, count, type);

    // Define connections between streams and kernels.
    // TODO: Fails when this is an odd # (something to do w/ how work is split).
    auto max_parallelization_factor = 1;
    auto tiles_per_cb = 4;
    current::Map map({&kernel_a, &kernel_b}, {&source0, &sink}, max_parallelization_factor, tiles_per_cb);
    map.add_connection(&source0, &kernel_a, "in0");
    map.add_connection(&kernel_a, "out0", &kernel_b, "in0");
    map.add_connection(&kernel_b, "out0", &sink);

    // Execute program.
    map.generate_device_kernels();
    map.execute();

    // Validate output.
    auto in = unpack_uint32_vec_into_bfloat16_vec(generator0_data);
    auto out = map.read_stream(&sink);
    auto out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(out);

    bool pass = true;
    for (size_t i = 0; i < count; i++) {
        bool close = is_close(in[i].to_float(), out_b16_vec[i].to_float());
        if (!close) {
            std::cout << i << ": in = " << in[i].to_float() << ", out = " << out_b16_vec[i].to_float() << "\n";
            std::cout << "\n";
        }
        pass &= close;
    }
    EXPECT_TRUE(pass);
}

TEST(CurrentTests, GatherTest) {
    // uint32_t num_indices = 1024 * 1024 * 256;
    uint32_t num_indices = 2048 * 16;
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
    std::vector<uint32_t> data_buffer = create_random_vector_of_bfloat16(data_buffer_size * 2, max_float, seed);
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
        // std::cout << "i: " << i << ",Index: " << index << ", ";
        auto input = in[index];
        // std::cout << "in[" << index << "] = " << input.to_float() << ", ";
        auto output = out_b16_vec[i];
        // std::cout << "out: " << out_b16_vec[i].to_float() << "\n";

        pass &= is_close(input.to_float(), output.to_float());
        // std::cout << "\n";
    }
    EXPECT_TRUE(pass);
}

TEST(CurrentTests, GatherTestSRAM) {
    // uint32_t num_indices = 1024 * 1024 * 256;
    uint32_t num_indices = 1024 * 1024;
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
    // std::vector<uint32_t> data_buffer = create_arange_vector_of_bfloat16(data_buffer_size * 2, false);
    std::vector<uint32_t> data_buffer = create_random_vector_of_bfloat16(data_buffer_size * 2, max_float, seed);
    // std::vector<uint32_t> data_buffer = create_random_vector_of_bfloat16(data_tile_size * 1024, max_float, seed);
    // std::vector<uint32_t> data_buffer = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles *
    // datum_size(type), 2.0F); std::vector<uint32_t> data_buffer = std::vector<uint32_t>(64, 0U); for (size_t i = 0; i
    // < data_buffer.size(); i++) {
    //     data_buffer[i] = i;
    // }
    std::uniform_int_distribution<uint32_t> dist(0, data_buffer_size - 1);
    auto index_vec = std::vector<uint32_t>(num_indices, 0);
    for (size_t i = 0; i < num_indices; i++) {
        index_vec[i] = dist(rng);
        // index_vec[i] = i;
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
    bool use_sram = true;
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
    auto in = unpack_uint32_vec_into_bfloat16_vec(data_buffer);
    auto out = map.read_stream(&sink);
    auto out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(out);

    bool pass = true;
    for (size_t i = 0; i < num_indices; i++) {
        auto index = index_vec[i];  // Since we're using SRAM, don't have to scale accesses.
        std::cout << std::dec;      // Force decimal output
        // std::cout << "i: " << i << ",Index: " << index << ", ";
        auto input = in[index];
        // std::cout << "in[" << index << "] = " << input.to_float() << ", ";
        auto output = out_b16_vec[i];
        // std::cout << "out: " << out_b16_vec[i].to_float() << "\n";

        pass &= is_close(input.to_float(), output.to_float());
        // std::cout << "\n";
    }
    EXPECT_TRUE(pass);
}

TEST(CurrentTests, GatherTestMultipleAccesses) {
    // uint32_t num_indices = 1024 * 1024 * 256;
    uint32_t num_indices = 2048 * 16;
    uint32_t data_buffer_n_elements = 1024 * 256;
    uint8_t accesses_per_token = 2;
    assert(num_indices % accesses_per_token == 0 && "Accesses per token must evenly divide indices!\n");
    auto type = tt::DataFormat::Float16_b;
    auto input_ntiles = static_cast<uint32_t>(std::ceil(num_indices / static_cast<double>(TILE_SIZE)));
    auto output_ntiles =
        static_cast<uint32_t>(std::ceil((num_indices / accesses_per_token) / static_cast<double>(TILE_SIZE)));
    std::cout << "input_ntiles: " << input_ntiles << "\n";
    std::cout << "output_ntiles: " << output_ntiles << "\n";

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
    std::vector<uint32_t> data_buffer = create_arange_vector_of_bfloat16(data_buffer_n_elements * 2, false);
    // std::vector<uint32_t> data_buffer = create_random_vector_of_bfloat16(data_buffer_n_elements * 2, max_float,
    // seed); std::vector<uint32_t> data_buffer = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles *
    // datum_size(type), 2.0F); std::vector<uint32_t> data_buffer = std::vector<uint32_t>(64, 0U); for (size_t i = 0; i
    // < data_buffer.size(); i++) {
    //     data_buffer[i] = i;
    // }
    std::uniform_int_distribution<uint32_t> dist(0, data_buffer_n_elements - 1);
    auto index_vec = std::vector<uint32_t>(num_indices, 0);
    std::cout << "Index vec: \n";
    for (size_t i = 0; i < num_indices / accesses_per_token; i++) {
        // auto idx = dist(rng);
        // index_vec[i * 2] = dist(rng);
        index_vec[i * 2] = i / 2;
        // std::cout << i * 2 << ": " << index_vec[i * 2] << "\n";
        // index_vec[i * 2 + 1] = dist(rng);
        index_vec[i * 2 + 1] = i / 2;
        // std::cout << i * 2 + 1 << ": " << index_vec[i * 2 + 1] << "\n";
    }

    // If accesseses_per_token > 1, then output_data.size() < num_indices
    auto output_data = create_constant_vector_of_bfloat16(data_tile_size * output_ntiles, -1.0F);

    // std::vector<uint32_t> generator1_data =
    //     create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), 4.0F);
    // std::vector<uint32_t> output_data = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 0.0f);

    // Init kernel.
    current::Kernel kernel_a;

    // Define ports and set compute kernel.
    kernel_a.add_input_port("in0", type);
    kernel_a.add_output_port("out0", type);

    kernel_a.set_compute_kernel(
        R"(
        out0 = in1;
    )",
        false);

    // Define streams.
    bool use_sram = false;
    current::GatherStream gather_stream(
        data_buffer, type, data_buffer_n_elements, index_vec, use_sram, accesses_per_token);
    current::Stream sink(output_data, num_indices / accesses_per_token, type);

    // Define connections between streams and kernels.
    auto max_parallelization_factor = 1;
    auto tiles_per_cb = 1;
    current::Map map({&kernel_a}, {&gather_stream, &sink}, max_parallelization_factor, tiles_per_cb);
    map.add_connection(&gather_stream, &kernel_a, "in0");
    map.add_connection(&kernel_a, "out0", &sink);

    // Execute program.
    map.propagate_counts();
    std::cout << "Propagated counts!\n";
    map.generate_device_kernels();
    std::cout << "Generated kernels!\n";
    map.execute();

    std::cout << "Finished!\n";

    // Validate output
    auto in_raw = map.read_gather_stream(&gather_stream, true);
    auto in = unpack_uint32_vec_into_bfloat16_vec(in_raw);
    auto out = map.read_stream(&sink);
    auto out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(out);
    std::cout << "Out size: " << out_b16_vec.size() << "\n";

    bool pass = true;
    // Since we have 2 accesses per token, we'll process indices in pairs
    // and output_size will be num_indices/2
    for (size_t i = 0; i < num_indices / accesses_per_token; i++) {
        // Get the two indices for this token
        auto index1 = index_vec[i * accesses_per_token] * 16;
        auto index2 = index_vec[i * accesses_per_token + 1] * 16;

        std::cout << std::dec;  // Force decimal output
        auto input1 = in[index1];
        auto input2 = in[index2];
        auto output = out_b16_vec[i];

        // Expected output is average of the two gathered values
        // float expected = (input1.to_float() + input2.to_float()) * 0.5f;
        float expected = input1.to_float();

        std::cout << "Token " << i << ": "
                  << "index1=" << index1 << " val1=" << input1.to_float() << ", "
                  << "index2=" << index2 << " val2=" << input2.to_float() << ", "
                  << "expected=" << expected << ", "
                  << "got=" << output.to_float() << "\n";

        pass &= is_close(expected, output.to_float());
    }
    EXPECT_TRUE(pass);
}

TEST(CurrentTests, GatherTestMultipleAccessesSRAM) {
    // uint32_t num_indices = 1024 * 1024 * 256;
    uint32_t num_indices = 1024 * 2 * 16;
    uint32_t data_buffer_n_elements = 1024 * 256;
    uint8_t accesses_per_token = 2;
    assert(num_indices % accesses_per_token == 0 && "Accesses per token must evenly divide indices!\n");
    auto type = tt::DataFormat::Float16_b;
    auto input_ntiles = static_cast<uint32_t>(std::ceil(num_indices / static_cast<double>(TILE_SIZE)));
    auto output_ntiles =
        static_cast<uint32_t>(std::ceil((num_indices / accesses_per_token) / static_cast<double>(TILE_SIZE)));
    std::cout << "input_ntiles: " << input_ntiles << "\n";
    std::cout << "output_ntiles: " << output_ntiles << "\n";

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
    // std::vector<uint32_t> data_buffer = create_arange_vector_of_bfloat16(data_buffer_n_elements * 2, false);
    std::vector<uint32_t> data_buffer = create_random_vector_of_bfloat16(data_buffer_n_elements * 2, max_float, seed);
    // std::vector<uint32_t> data_buffer = create_constant_vector_of_bfloat16(data_buffer_n_elements * 2, 2.0F);
    // std::vector<uint32_t> data_buffer = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles *
    // datum_size(type), 2.0F); std::vector<uint32_t> data_buffer = std::vector<uint32_t>(64, 0U); for (size_t i = 0; i
    // < data_buffer.size(); i++) {
    //     data_buffer[i] = i;
    // }
    std::uniform_int_distribution<uint32_t> dist(0, data_buffer_n_elements - 1);
    auto index_vec = std::vector<uint32_t>(num_indices, 0);
    std::cout << "Index vec: \n";
    for (size_t i = 0; i < num_indices / accesses_per_token; i++) {
        // auto idx = dist(rng);
        index_vec[i * 2] = dist(rng);
        // index_vec[i * 2] = i / 2;
        // std::cout << i * 2 << ": " << index_vec[i * 2] << "\n";
        index_vec[i * 2 + 1] = dist(rng);
        // index_vec[i * 2 + 1] = i / 2;
        // std::cout << i * 2 + 1 << ": " << index_vec[i * 2 + 1] << "\n";
    }

    // If accesseses_per_token > 1, then output_data.size() < num_indices
    auto output_data = create_constant_vector_of_bfloat16(data_tile_size * output_ntiles, -1.0F);

    // std::vector<uint32_t> generator1_data =
    //     create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * datum_size(type), 4.0F);
    // std::vector<uint32_t> output_data = create_constant_vector_of_bfloat16(TILE_SIZE * n_tiles * 2, 0.0f);

    // Init kernel.
    current::Kernel kernel_a;

    // Define ports and set compute kernel.
    kernel_a.add_input_port("in0", type);
    kernel_a.add_output_port("out0", type);

    kernel_a.set_compute_kernel(
        R"(
        out0 = (in0 + in1) * 0.5;
    )",
        false);

    // Define streams.
    bool use_sram = true;
    current::GatherStream gather_stream(
        data_buffer, type, data_buffer_n_elements, index_vec, use_sram, accesses_per_token);
    current::Stream sink(output_data, num_indices / accesses_per_token, type);

    // Define connections between streams and kernels.
    auto max_parallelization_factor = 1;
    auto tiles_per_cb = 1;
    current::Map map({&kernel_a}, {&gather_stream, &sink}, max_parallelization_factor, tiles_per_cb);
    map.add_connection(&gather_stream, &kernel_a, "in0");
    map.add_connection(&kernel_a, "out0", &sink);

    // Execute program.
    map.propagate_counts();
    std::cout << "Propagated counts!\n";
    map.generate_device_kernels();
    std::cout << "Generated kernels!\n";
    map.execute();

    std::cout << "Finished!\n";

    // Validate output
    // auto in_raw = map.read_gather_stream(&gather_stream, true);
    auto in = unpack_uint32_vec_into_bfloat16_vec(data_buffer);
    auto out = map.read_stream(&sink);
    auto out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(out);

    bool pass = true;
    // Since we have 2 accesses per token, we'll process indices in pairs
    // and output_size will be num_indices/2
    std::cout << "Checking output...\n";
    for (size_t i = 0; i < num_indices / accesses_per_token; i++) {
        // Get the two indices for this token
        auto index1 = index_vec[i * accesses_per_token];
        auto index2 = index_vec[i * accesses_per_token + 1];

        std::cout << std::dec;  // Force decimal output
        auto input1 = in[index1];
        auto input2 = in[index2];
        auto output = out_b16_vec[i];

        // Expected output is average of the two gathered values
        float expected = (input1.to_float() + input2.to_float()) * 0.5f;

        std::cout << "Token " << i << ": "
                  << "index1=" << index1 << " val1=" << input1.to_float() << ", "
                  << "index2=" << index2 << " val2=" << input2.to_float() << ", "
                  << "expected=" << expected << ", "
                  << "got=" << output.to_float() << "\n";

        pass &= is_close(expected, output.to_float());
    }
    EXPECT_TRUE(pass);

    // // Validate output.
    // auto in_raw = map.read_gather_stream(&gather_stream, true);
    // auto in = unpack_uint32_vec_into_bfloat16_vec(in_raw);
    // auto out = map.read_stream(&sink);
    // auto out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(out);

    // bool pass = true;
    // for (size_t i = 0; i < num_indices; i++) {
    //     auto index = index_vec[i] * 16;
    //     std::cout << std::dec;  // Force decimal output
    //     // std::cout << "i: " << i << ",Index: " << index << ", ";
    //     auto input = in[index];
    //     // std::cout << "in[" << index << "] = " << input.to_float() << ", ";
    //     auto output = out_b16_vec[i];
    //     // std::cout << "out: " << out_b16_vec[i].to_float() << "\n";

    //     pass &= is_close(input.to_float(), output.to_float());
    //     // std::cout << "\n";
    // }
    // EXPECT_TRUE(pass);
}
