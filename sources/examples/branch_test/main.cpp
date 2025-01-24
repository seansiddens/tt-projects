#include <chrono>
#include <cstdlib>

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

    // auto output_dram_data = create_constant_vector_of_bfloat16(dram_config.size, -1.0F);
    // EnqueueWriteBuffer(cq, output_dram, output_dram_data, true);

    // /* Use L1 circular buffers to set input buffers */
    // constexpr uint32_t src0_cb_index = CB::c_in0;
    // CircularBufferConfig cb_src0_config =
    //     CircularBufferConfig(b16_tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
    //         .set_page_size(src0_cb_index, b16_tile_size_bytes);
    // CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    auto l1_size = device->l1_size_per_core();
    tt::log_info("L1 size: {}:", l1_size);
    uint32_t max_l1_size = l1_size / 2 - L1_UNRESERVED_BASE;
    tt::log_info("Max L1 size: {}", max_l1_size);

    InterleavedBufferConfig sram_config{
        .device = device, .size = max_l1_size, .page_size = max_l1_size, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> sram_buf = CreateBuffer(sram_config);
    auto sram_addr = sram_buf->address();
    std::vector<uint32_t> input_dram_data(sram_config.size / sizeof(uint32_t));
    for (size_t i = 0; i < input_dram_data.size(); i++) {
        input_dram_data[i] = rand();
    }
    tt::tt_metal::detail::WriteToDeviceL1(device, core, sram_buf->address(), input_dram_data);
    tt::log_info("Wrote {} bytes to SRAM on core {}", input_dram_data.size() * sizeof(uint32_t), core);

    /* Specify data movement kernel for reading/writing data to/from DRAM */
    // TODO: Test all processors?
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "sources/examples/branch_test/kernels/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Configure program and runtime kernel arguments, then execute */
    std::vector<uint32_t> iterations = {1000, 10000, 100000, 1000000};
    std::vector<std::chrono::steady_clock::duration> durations;

    for (size_t i = 0; i < iterations.size(); i++) {
        SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {iterations[i], sram_addr, static_cast<uint32_t>(input_dram_data.size())});
        tt_metal::detail::CompileProgram(device, program);
        auto start = std::chrono::steady_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto end = std::chrono::steady_clock::now();
        durations.push_back(end - start);
    }

    // Print results.
    for (size_t i = 0; i < iterations.size(); i++) {
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(durations[i]);
        std::cout << iterations[i] << " iterations: " << duration_us << "\n";
    }

    CloseDevice(device);
}
