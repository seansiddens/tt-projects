#include <fstream>
#include <vector>

#include "common/logger.hpp"
#include "host_api.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "impl/buffers/buffer.hpp"
#include "impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::tt_metal;

// Function to dump a vector<uint32_t> to a binary file
void dump_vector_to_binary(const std::vector<uint32_t> &vec, const std::string &filename) {
    // Open the file in binary mode for writing
    std::ofstream out_file(filename, std::ios::out | std::ios::binary);
    if (!out_file) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write the raw data of the vector to the file
    out_file.write(reinterpret_cast<const char *>(vec.data()), vec.size() * sizeof(uint32_t));

    // Check if the write operation was successful
    if (!out_file) {
        std::cerr << "Error: Failed to write data to file " << filename << "." << std::endl;
    }

    // Close the file
    out_file.close();
}

int main(int argc, char **argv) {
    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    auto l1_size = device->l1_size_per_core();
    tt::log_info("L1 size: {}:", l1_size);

    // Addresses 0 ~ L1_UNRESERVED_BASE are reserved.
    tt::log_info("L1_UNRESERVED_BASE = {}", L1_UNRESERVED_BASE);

    const uint32_t CANARY_VAL = 0xCAFEBABE;

    // Output DRAM buffer for storing dump of SRAM.
    InterleavedBufferConfig dram_config{
        .device = device, .size = l1_size, .page_size = l1_size, .buffer_type = BufferType::DRAM};

    std::shared_ptr<Buffer> output_dram = CreateBuffer(dram_config);
    uint32_t output_dram_address = output_dram->address();
    auto output_dram_coords = output_dram->noc_coordinates();
    uint32_t output_dram_x = output_dram_coords.x;
    uint32_t output_dram_y = output_dram_coords.y;

    std::vector<uint32_t> initial_data(l1_size / sizeof(uint32_t), 10);
    EnqueueWriteBuffer(cq, output_dram, initial_data, true);
    tt::log_info("Initialized output buffer with values.");

    KernelHandle kernel = CreateKernel(
        program,
        "sources/examples/leftover_read/kernels/reader.cpp",
        core,
        // TODO: Try other processors? Would that really matter though?
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(program, kernel, core, {output_dram_address, output_dram_x, output_dram_y, l1_size});

    EnqueueProgram(cq, program, true);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, output_dram, result_vec, true);

    auto filename = "output.bin";
    dump_vector_to_binary(result_vec, filename);
    tt::log_info("Dumped data to {}", filename);

    CloseDevice(device);
}
