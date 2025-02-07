#include "dataflow_api.h"

// constexpr uint32_t val = (0x40e0 << 16) | 0x40e0;
constexpr uint16_t val = 0xbf80;

void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    // DPRINT << "src addr: " << src0_dram << "\n";
    // uint16_t *foo = (uint16_t *)src0_dram;
    // *foo = 0;
    // DPRINT << "src[0]: " << float(foo[0]) << "\n";
    uint32_t dst_dram = get_arg_val<uint32_t>(1);
    uint32_t src0_dram_noc_x = get_arg_val<uint32_t>(2);
    uint32_t src0_dram_noc_y = get_arg_val<uint32_t>(3);
    uint32_t dst_dram_noc_x = get_arg_val<uint32_t>(4);
    uint32_t dst_dram_noc_y = get_arg_val<uint32_t>(5);
    uint32_t in0_addr = get_arg_val<uint32_t>(6);
    uint32_t index_dram_noc_x = get_arg_val<uint32_t>(7);
    uint32_t index_dram_noc_y = get_arg_val<uint32_t>(8);
    uint32_t in0_ntiles = get_arg_val<uint32_t>(9);
    const InterleavedAddrGenFast<true> in0_addr_gen = {
        .bank_base_address = in0_addr,
        .page_size = 4096,
        .data_format = DataFormat::UInt32,
    };

    // NoC coords (x,y) depending on DRAM location on-chip
    uint64_t src0_dram_noc_addr = get_noc_addr(src0_dram_noc_x, src0_dram_noc_y, src0_dram);
    // uint64_t src1_dram_noc_addr = get_noc_addr(src1_dram_noc_x, src1_dram_noc_y, src1_dram);
    uint64_t dst_dram_noc_addr = get_noc_addr(dst_dram_noc_x, dst_dram_noc_y, dst_dram);
    uint64_t index_dram_noc_addr = get_noc_addr(index_dram_noc_x, index_dram_noc_y, in0_addr);

    constexpr uint32_t in0 = tt::CB::c_in0;        // index=0
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;  // index=1

    // single-tile ublocks
    uint32_t data_tile_size = get_tile_size(in0);
    uint32_t index_tile_size = get_tile_size(cb_id_in1);
    // DPRINT << "[READER] DATA TILE SIZE: " << data_tile_size;
    // DPRINT << "[READER] INDEX TILE SIZE: " << index_tile_size;

    uint32_t l1_write_addr_data = get_write_ptr(in0);
    uint32_t in0_indices_write_ptr = get_write_ptr(cb_id_in1);

    // Read data from DRAM -> L1 circular buffers
    // noc_async_read(src0_dram_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
    // noc_async_read_barrier();
    // noc_async_read(src1_dram_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
    // noc_async_read_barrier();
    // uint16_t *sram_ptr = (uint16_t *)src0_dram_noc_addr;

    for (uint32_t i = 0; i < in0_ntiles; i++) {
        // DPRINT << "tile " << i << "\n";
        // Read tile of index data from DRAM -> L1 circular buffer.
        uint32_t data_buf_offset = i * data_tile_size;  // Offset into data DRAM buf for our current tile.
        // DPRINT << "Data buf offset: " << data_buf_offset << "\n";
        uint32_t index_buf_offset = i * index_tile_size;  // Offset into idx DRAM buf for our current tile.
        // noc_async_read(index_dram_noc_addr + index_buf_offset, l1_write_addr_index, index_tile_size);
        noc_async_read_tile(i, in0_addr_gen, in0_indices_write_ptr);
        noc_async_read_barrier();

        uint32_t *dat1 = (uint32_t *)in0_indices_write_ptr;
        uint16_t *cb_ptr = (uint16_t *)l1_write_addr_data;
        uint16_t *sram_ptr = (uint16_t *)src0_dram;
        for (int j = 0; j < 1024; j++) {
            cb_ptr[j] = val;
        }

        for (int j = 0; j < 1024; j++) {
            uint32_t index = dat1[j];
            // DPRINT << "Index: " << index << "\n";
            // `index` is an index into the global input data DRAM buffer, we need to scale by the size of the data type
            // (sizeof(bfloat16) == 2)
            // DPRINT << "i: " << i * 1024 + j << ", index: " << index << "\n";
            uint32_t index_offset = index * 2 * 16;  // 32 byte aligned.
            uint32_t l1_offset =
                j * 2;  // Need to correclty index into the L1 where we are storing the data values gathered.
            // noc_async_read(src0_dram_noc_addr + index_offset, l1_write_addr_data + l1_offset, 2);

            // Read data directly from L1, no scaling required.
            // *(l1_write_addr_data + l1_offset) = *(src0_dram + index);
            cb_ptr[j] = sram_ptr[index];
        }
        noc_async_read_barrier();

        // Write data from L1 circulr buffer (in0) -> DRAM
        noc_async_write(l1_write_addr_data, dst_dram_noc_addr + data_buf_offset, data_tile_size);
        noc_async_write_barrier();
    }
}
