#include <chrono>

#include "common/bfloat16.hpp"
#include "common/logger.hpp"
#include "host_api.hpp"
#include "impl/buffers/buffer.hpp"
#include "impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    uint32_t num_tiles = 1;
    constexpr uint32_t tile_size = 1024;
    constexpr uint32_t b16_tile_size_bytes = 2 * tile_size;

    auto l1_size = device->l1_size_per_core();
    tt::log_info("L1 size: {}:", l1_size);
    uint32_t max_l1_size = l1_size / 2 - L1_UNRESERVED_BASE;
    tt::log_info("Max L1 size: {}", max_l1_size);

    InterleavedBufferConfig dram_config{
        .device = device,
        .size = b16_tile_size_bytes * num_tiles,
        .page_size = b16_tile_size_bytes * num_tiles,
        .buffer_type = BufferType::DRAM};

    InterleavedBufferConfig sram_config{
        .device = device, .size = max_l1_size, .page_size = max_l1_size, .buffer_type = BufferType::L1};

    std::shared_ptr<Buffer> input_dram = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> output_dram = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> sram_buf = CreateBuffer(sram_config);
    uint32_t input_dram_address = input_dram->address();
    auto input_dram_coords = input_dram->noc_coordinates();
    uint32_t input_dram_x = input_dram_coords.x;
    uint32_t input_dram_y = input_dram_coords.y;
    uint32_t output_dram_address = output_dram->address();
    auto output_dram_coords = output_dram->noc_coordinates();
    uint32_t output_dram_x = output_dram_coords.x;
    uint32_t output_dram_y = output_dram_coords.y;

    /* Create source data and write to DRAM */
    // std::cout << "Index vec:\n";
    // std::vector<uint32_t> index_vec(num_indices, 0);
    // for (size_t i = 0; i < index_vec.size(); i++) {
    //     index_vec[i] = i;
    // }
    // index_vec[0] = 2;
    // index_vec[1] = 2;
    // index_vec[2] = 2;
    // index_vec[3] = 2;
    // auto rng = std::mt19937{std::random_device{}()};  // or use time-based seed
    // std::shuffle(index_vec.begin(), index_vec.end(), rng);
    // for (auto val : index_vec) {
    //     std::cout << val << " ";
    // }
    // std::cout << "\n";

    // std::vector<uint32_t> src0_vec(1, 14);
    // std::cout << "Data buffer size: " << input_data_sram_config.size << "\n";
    // std::cout << "Src 0 vec:\n";
    // std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(dram_config.size, false);
    // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    float canary_val = 564.0F;
    std::vector<uint32_t> input_dram_data = create_constant_vector_of_bfloat16(sram_config.size, canary_val);
    // std::vector<uint32_t> input_dram_data = create_random_vector_of_bfloat16(dram_config.size, 10, seed);
    // std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(data_dram_config.size, false);
    // std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(data_dram_config.size, -1.0F);
    // std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(input_data_sram_config.size, false);
    // auto in = unpack_uint32_vec_into_bfloat16_vec(src0_vec);
    // for (size_t i = 0; i < src0_vec.size(); i++) {
    //     src0_vec[i] = 0;
    // }
    // std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(input_data_sram_config.size, 0.0F);
    // std::vector<uint32_t> src0_vec(input_data_sram_config.size / 4, 0);
    // std::vector<bfloat16> in = unpack_uint32_vec_into_bfloat16_vec(src0_vec);
    // std::vector<bfloat16> in(input_data_dram_config.size / 2, bfloat16(-1.0F));
    // // float count = 0.0F;
    // for (size_t i = 0; i < in.size(); i++) {
    //     in[i] = bfloat16(float(i / 16));
    // }
    // std::vector<uint32_t> src0_vec = pack_bfloat16_vec_into_uint32_vec(in);
    // std::cout << "Input vec size: " << in.size() << "\n";
    // for (auto val : in) {
    //     std::cout << val.to_float() << " ";
    // }
    // std::cout << std::endl;

    // EnqueueWriteBuffer(cq, input_dram, input_dram_data, true);
    // // EnqueueWriteBuffer(cq, src0_buffer, src0_vec, true);

    // tt::tt_metal::detail::WriteToDeviceL1(device, core, sram_buf->address(), input_dram_data);
    // tt::log_info("Wrote {} bytes to SRAM on core {}", input_dram_data.size() * sizeof(uint32_t), core);

    std::vector<uint32_t> out_data;
    tt::tt_metal::detail::ReadFromDeviceL1(device, core, sram_buf->address(), sram_config.size, out_data);
    std::vector<bfloat16> out = unpack_uint32_vec_into_bfloat16_vec(out_data);
    for (size_t i = 0; i < sram_config.size / 2; i++) {
        std::cout << i << ": " << out[i].to_float() << "\n";
    }

    // auto output_dram_data = create_constant_vector_of_bfloat16(dram_config.size, -1.0F);
    // EnqueueWriteBuffer(cq, output_dram, output_dram_data, true);

    // /* Use L1 circular buffers to set input buffers */
    // constexpr uint32_t src0_cb_index = CB::c_in0;
    // CircularBufferConfig cb_src0_config =
    //     CircularBufferConfig(b16_tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
    //         .set_page_size(src0_cb_index, b16_tile_size_bytes);
    // CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    /* Specify data movement kernel for reading/writing data to/from DRAM */
    // KernelHandle binary_reader_kernel_id = CreateKernel(
    //     program,
    //     "sources/examples/leftover_write/kernels/reader.cpp",
    //     core,
    //     DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // /* Configure program and runtime kernel arguments, then execute */
    // SetRuntimeArgs(
    //     program,
    //     binary_reader_kernel_id,
    //     core,
    //     {
    //         input_dram_address,
    //         input_dram_x,
    //         input_dram_y,
    //         output_dram_address,
    //         output_dram_x,
    //         output_dram_y,
    //         num_tiles
    //     });

    // EnqueueProgram(cq, program, true);
    // Finish(cq);

    // /* Read in result into a host vector */
    // std::vector<uint32_t> result_vec;
    // EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    // std::vector<bfloat16> out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    // std::cout << "Output vec size: " << out_b16_vec.size() << "\n";

    // std::cout << "Bfloat data:\n";
    // for (auto val : out_b16_vec) {
    //     std::cout << val.to_float() << " ";
    // }

    // std::cout << "Result: \n";
    // for (size_t i = 0; i < num_indices; i++) {
    //     std::cout << "i: " << i << ", in[i]: " << in[i].to_float() << ", " << "out[i]: " << out_b16_vec[i].to_float()
    //     << "\n";
    // }
    // std::cout << std::endl;

    // Validate output of gather operation.
    // out[i] == in[idx[i]]
    // for (size_t i = 0; i < num_indices; i++) {
    //     auto index = index_vec[i];
    //     std::cout << "i: " << i << ", Index: " << index << ", ";
    //     auto input = in[index];
    //     std::cout << "in[" << index << "] = " << input.to_float() << ", ";
    //     auto output = out_b16_vec[i];
    //     std::cout << "out: " << out_b16_vec[i].to_float() << "\n";

    //     is_close(input.to_float(), output.to_float());
    //     // std::cout << "\n";
    // }

    CloseDevice(device);
}
