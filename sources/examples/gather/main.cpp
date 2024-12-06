#include <chrono>

#include "common/bfloat16.hpp"
#include "host_api.hpp"
#include "impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_size = 1024;
    constexpr uint32_t b16_tile_size = 2 * tile_size;
    constexpr uint32_t u32_tile_size = 4 * tile_size;

    uint32_t num_indices = 1024;
    std::cout << "Number of indices: " << num_indices << "\n";
    uint32_t index_ntiles = std::ceil(static_cast<float>(num_indices) / tile_size);
    std::cout << "index_ntiles: " << index_ntiles << "\n";
    uint32_t data_ntiles = std::ceil(static_cast<float>(num_indices) / tile_size);
    std::cout << "data_ntiles: " << data_ntiles << "\n";

    InterleavedBufferConfig data_dram_config{
        .device = device,
        .size = b16_tile_size * data_ntiles,
        .page_size = b16_tile_size,
        .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig index_dram_config{
        .device = device,
        .size = u32_tile_size * index_ntiles,
        .page_size = u32_tile_size,
        .buffer_type = BufferType::DRAM};

    std::shared_ptr<Buffer> index_buffer = CreateBuffer(index_dram_config);
    std::shared_ptr<Buffer> src0_dram_buffer = CreateBuffer(data_dram_config);
    std::shared_ptr<Buffer> src1_dram_buffer = CreateBuffer(data_dram_config);
    std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(data_dram_config);

    auto src0_dram_noc_coord = src0_dram_buffer->noc_coordinates();
    auto src1_dram_noc_coord = src1_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
    auto index_dram_noc_coord = index_buffer->noc_coordinates();
    uint32_t src0_dram_noc_x = src0_dram_noc_coord.x;
    uint32_t src0_dram_noc_y = src0_dram_noc_coord.y;
    uint32_t src1_dram_noc_x = src1_dram_noc_coord.x;
    uint32_t src1_dram_noc_y = src1_dram_noc_coord.y;
    uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;
    uint32_t index_dram_noc_x = index_dram_noc_coord.x;
    uint32_t index_dram_noc_y = index_dram_noc_coord.y;

    /* Create source data and write to DRAM */
    std::cout << "Index vec:\n";
    std::vector<uint32_t> index_vec(num_indices, 0);
    for (size_t i = 0; i < index_vec.size(); i++) {
        index_vec[i] = 16;
    }
    // index_vec[0] = 0;
    // index_vec[1] = 0;
    // auto rng = std::mt19937{std::random_device{}()};  // or use time-based seed
    // std::shuffle(index_vec.begin(), index_vec.end(), rng);
    for (auto val : index_vec) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // std::vector<uint32_t> src0_vec(1, 14);
    std::cout << "Src 0 vec:\n";
    // std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(dram_config.size, false);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    // std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(data_dram_config.size, 10, seed);
    std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(data_dram_config.size, false);
    std::vector<bfloat16> in_b16_vec = unpack_uint32_vec_into_bfloat16_vec(src0_vec);
    std::cout << "Input vec size: " << in_b16_vec.size() << "\n";
    for (auto val : in_b16_vec) {
        std::cout << val.to_float() << " ";
    }
    std::cout << std::endl;
    std::vector<uint32_t> src1_vec(1, 7);

    EnqueueWriteBuffer(cq, index_buffer, index_vec, true);
    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, true);
    EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, true);
    auto dst_initial_data = create_constant_vector_of_bfloat16(data_dram_config.size, -1.0F);
    EnqueueWriteBuffer(cq, dst_dram_buffer, dst_initial_data, true);

    /* Use L1 circular buffers to set input buffers */
    constexpr uint32_t src0_cb_index = CB::c_in0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(b16_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, b16_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = CB::c_in1;
    CircularBufferConfig cb_src1_config = CircularBufferConfig(1024 * 4, {{src1_cb_index, tt::DataFormat::UInt32}})
                                              .set_page_size(src1_cb_index, 1024 * 4);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    /* Specify data movement kernel for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "sources/examples/gather/kernels/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {
            src0_dram_buffer->address(),
            src1_dram_buffer->address(),
            dst_dram_buffer->address(),
            src0_dram_noc_x,
            src0_dram_noc_y,
            src1_dram_noc_x,
            src1_dram_noc_y,
            dst_dram_noc_x,
            dst_dram_noc_y,
            index_buffer->address(),
            index_dram_noc_x,
            index_dram_noc_y,
            index_ntiles,
        });

    EnqueueProgram(cq, program, true);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    std::vector<bfloat16> out_b16_vec = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    std::cout << "Output vec size: " << out_b16_vec.size() << "\n";

    std::cout << "Bfloat data:\n";
    for (auto val : out_b16_vec) {
        std::cout << val.to_float() << " ";
    }
    std::cout << std::endl;

    // Validate output of gather operation.
    // out[i] == in[idx[i]]
    for (size_t i = 0; i < num_indices; i++) {
        auto index = index_vec[i];
        std::cout << "Index: " << index << ", ";
        auto input = in_b16_vec[index];
        std::cout << "in[" << index << "] = " << input.to_float() << ", ";
        auto output = out_b16_vec[i];
        std::cout << "out: " << out_b16_vec[i].to_float() << "\n";

        is_close(input.to_float(), output.to_float());
        std::cout << "\n";
    }

    CloseDevice(device);
}
