// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <cstdlib>

#include "dataflow_api.h"

volatile int count = 0;

void kernel_main() {
    uint32_t iterations = get_arg_val<uint32_t>(0);
    uint32_t buf_addr = get_arg_val<uint32_t>(1);
    uint32_t buf_size = get_arg_val<uint32_t>(2);

    for (uint32_t i = 0; i < iterations; i++) {
        // Predictable branch
        if (i < iterations) {
            count += 1;
        }
        // Unpredictable branch
        // if (rand() % 2 == 0) {
        //     count += 1;
        // }
    }
}
