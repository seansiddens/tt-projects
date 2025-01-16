
#include <gtest/gtest.h>

#include <cstdint>
#include <ctime>
#include <random>

#include "common.hpp"
#include "common/bfloat16.hpp"
#include "common/tt_backend_api_types.hpp"
#include "map.hpp"
#include "stream.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

TEST(CurrentTests, B16EltwiseSAXPY) {
    uint32_t count = 1024 * 512;
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

// Performs a horizontal-only box blur on a single-channel float array
// Out-of-bounds pixels are treated as 0.0
// kernel_size must be odd
std::vector<bfloat16> horizontalBoxBlur(const std::vector<bfloat16> &data, int width, int height, int kernel_size) {
    std::vector<bfloat16> result(width * height);

    // Half kernel size for easier boundary calculations
    int half_k = kernel_size / 2;

    // For each row
    for (int y = 0; y < height; y++) {
        // For each pixel in the row
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            // For each kernel element horizontally
            for (int kx = -half_k; kx <= half_k; kx++) {
                int px = x + kx;

                // If inside bounds, add pixel value; if outside, add 0.0
                if (px >= 0 && px < width) {
                    sum += data[y * width + px].to_float();
                }
            }

            // Always divide by full kernel size, including out-of-bounds positions
            result[y * width + x] = bfloat16(sum / kernel_size);
        }
    }

    return result;
}

void create_test_image() {
    const int width = 512;
    const int height = 512;
    const int channels = 1;  // RGB

    std::vector<unsigned char> image(width * height * channels, 255);  // White background

    // Draw a black cross
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (x == width / 2 || y == height / 2) {
                int idx = (y * width + x) * channels;
                image[idx] = 0;  // R
                // image[idx + 1] = 0; // G
                // image[idx + 2] = 0; // B
            }
        }
    }

    // Draw a checkerboard pattern in one quarter
    int square_size = 32;
    for (int x = 0; x < width / 2; x++) {
        for (int y = 0; y < height / 2; y++) {
            if (((x / square_size) + (y / square_size)) % 2 == 0) {
                int idx = (y * width + x) * channels;
                image[idx] = 0;  // R
                // image[idx + 1] = 0; // G
                // image[idx + 2] = 0; // B
            }
        }
    }

    // Draw some concentric circles in another quarter
    for (int x = width / 2; x < width; x++) {
        for (int y = height / 2; y < height; y++) {
            int dx = x - width * 3 / 4;
            int dy = y - height * 3 / 4;
            int dist = (int)sqrt(dx * dx + dy * dy);
            if (dist % 32 < 16) {
                int idx = (y * width + x) * channels;
                image[idx] = 0;  // R
                // image[idx + 1] = 0; // G
                // image[idx + 2] = 0; // B
            }
        }
    }

    stbi_write_png("test_image.png", width, height, channels, image.data(), width * channels);
}

TEST(CurrentTests, BoxFilter) {
    bool pass = true;
    int width, height, channels;
    unsigned char *image = stbi_load("test_image.png", &width, &height, &channels, 1);
    std::cout << "Loaded input image! Width: " << width << ", height: " << height << ", original channels: " << channels
              << "\n";

    auto kernel_size = 3;

    std::vector<bfloat16> input_image_data(width * height);
    for (int i = 0; i < width * height; i++) {
        input_image_data[i] = image[i] / 255.0f;
    }

    // Baseline Implementation
    auto baseline_result = horizontalBoxBlur(input_image_data, width, height, kernel_size);

    // Convert back to unsigned char
    std::vector<unsigned char> output_baseline_data(width * height);
    for (int i = 0; i < width * height; i++) {
        output_baseline_data[i] = static_cast<unsigned char>(baseline_result[i].to_float() * 255.0f + 0.5f);
    }

    // Save the result
    int success = stbi_write_png("out_baseline.png", width, height, 1, output_baseline_data.data(), width);

    if (!success) {
        std::cout << "Failed to write output image!\n";
        pass &= false;
    }
    stbi_image_free(image);
    std::cout << "Successfully processed baseline image\n";

    // Current Implementation.
    auto image_data = pack_bfloat16_vec_into_uint32_vec(input_image_data);
    uint8_t accesses_per_token =
        kernel_size + 1;  // TODO: Fix it so that access per token doesnt need to evenly divide tiles.
    uint32_t num_indices = width * height * accesses_per_token;  // Two accesses per pixel.
    uint32_t data_buffer_n_elements = width * height;
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

    // Init kernel.
    current::Kernel kernel_a;

    // Define ports and set compute kernel.
    kernel_a.add_input_port("in0", type);
    kernel_a.add_output_port("out0", type);

    kernel_a.set_compute_kernel(
        R"(
        out0 = (in0 + in1 + in2) * 0.33;
    )",
        false);

    // Define streams.
    bool use_sram = true;

    // Index vector.
    std::vector<uint32_t> index_vec(num_indices, 0);
    auto index = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // For edge cases we are clamping to the edge (might not be same logic as baseline).
            // Generate indices for our kernel.
            // Left pixel.
            index_vec[index] = y * width + (x > 0 ? x - 1 : 0);
            index++;
            // Center pixel
            index_vec[index] = y * width + x;
            index++;
            // Right pixel
            index_vec[index] = y * width + (x < width - 1 ? x + 1 : width - 1);
            index++;

            // Unsed index to make evenly divisible.
            index_vec[index] = 0;
            index++;
        }
    }
    current::GatherStream gather_stream(
        image_data, type, data_buffer_n_elements, index_vec, use_sram, accesses_per_token);
    std::cout << "Created gather stream!\n";

    // Output stream.
    std::vector<uint32_t> out_data(data_buffer_n_elements, 0);
    current::Stream sink(out_data, data_buffer_n_elements, type);
    std::cout << "Created output stream!\n";

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

    //  // Validate output
    // auto in_raw = map.read_gather_stream(&gather_stream, true);
    // auto in = unpack_uint32_vec_into_bfloat16_vec(in_raw);
    auto out = map.read_stream(&sink);
    auto out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(out);
    std::cout << "Out size: " << out_b16_vec.size() << "\n";

    // for (size_t i = 0; i < data_buffer_n_elements; i++) {
    //     std::cout << i << ": " << out_b16_vec[i].to_float() << "\n";
    // }
    // // Since we have 2 accesses per token, we'll process indices in pairs
    // // and output_size will be num_indices/2
    // for (size_t i = 0; i < num_indices/accesses_per_token; i++) {
    //     // Get the two indices for this token
    //     auto index1 = index_vec[i * accesses_per_token] * 16;
    //     auto index2 = index_vec[i * accesses_per_token + 1] * 16;

    //     std::cout << std::dec;  // Force decimal output
    //     auto input1 = in[index1];
    //     auto input2 = in[index2];
    //     auto output = out_b16_vec[i];

    //     // Expected output is average of the two gathered values
    //     float expected = (input1.to_float() + input2.to_float()) * 0.5f;

    //     std::cout << "Token " << i << ": "
    //               << "index1=" << index1 << " val1=" << input1.to_float() << ", "
    //               << "index2=" << index2 << " val2=" << input2.to_float() << ", "
    //               << "expected=" << expected << ", "
    //               << "got=" << output.to_float() << "\n";

    //     pass &= is_close(expected, output.to_float());
    // }

    // Convert output to an image.
    std::vector<unsigned char> output_data(width * height);
    for (int i = 0; i < width * height; i++) {
        output_data[i] = static_cast<unsigned char>(out_b16_vec[i].to_float() * 255.0f + 0.5f);
        // std::cout << static_cast<uint8_t>(output_data[i]) << "\n";
    }

    // Save the result
    std::cout << "Saving output image...\n";
    success = stbi_write_png("out.png", width, height, 1, output_data.data(), width);

    if (!success) {
        std::cout << "Failed to write output image!\n";
        pass &= false;
    }
    std::cout << "Successfully processed output image\n";

    EXPECT_TRUE(pass);
}