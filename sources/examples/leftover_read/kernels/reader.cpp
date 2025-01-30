#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
    uint32_t l1_size = get_arg_val<uint32_t>(3);

    uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);
    noc_async_write(0, dst_noc_addr, l1_size);
}
