#include <iostream>

#include "host_api.hpp"
#include "impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(void) {
    Device *device = CreateDevice(0);
    std::cout << "Device created!\n";

    CloseDevice(device);
}
