#include <chrono>

#include "common/bfloat16.hpp"
#include "common/logger.hpp"
#include "host_api.hpp"
#include "impl/buffers/buffer.hpp"
#include "impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::tt_metal;

// Function to encode a message into a uint32_t buffer
std::vector<uint32_t> encode_message(const std::string &message) {
    std::vector<uint32_t> buffer;

    // Encode the message into uint32_t values
    for (size_t i = 0; i < message.size(); i += 4) {
        uint32_t value = 0;
        for (size_t j = 0; j < 4; ++j) {
            if (i + j < message.size()) {
                value |= static_cast<uint32_t>(message[i + j]) << (8 * j);
            }
        }
        buffer.push_back(value);
    }

    return buffer;
}

int main(int argc, char **argv) {
    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    std::vector<uint32_t> msg_data = encode_message("hello world!");
    auto message_size = msg_data.size() * sizeof(uint32_t);

    InterleavedBufferConfig sram_config{
        .device = device, .size = message_size, .page_size = message_size, .buffer_type = BufferType::L1};

    std::shared_ptr<Buffer> msg_buffer = CreateBuffer(sram_config);

    tt::tt_metal::detail::WriteToDeviceL1(device, core, msg_buffer->address(), msg_data);
    tt::log_info("Wrote {} bytes to SRAM on core {}", message_size, core);
    tt::log_info("message buffer address: {}", msg_buffer->address());

    CloseDevice(device);
}
