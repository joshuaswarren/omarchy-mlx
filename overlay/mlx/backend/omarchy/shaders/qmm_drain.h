#ifndef OMARCHY_QMM_DRAIN_H
#define OMARCHY_QMM_DRAIN_H

uint qmm_drain_row(uint subgroup, uint block, uint blocks_per_subgroup) {
  return (subgroup * blocks_per_subgroup + block) * 8u;
}

uint qmm_drain_slot(uint subgroup, uint column_block, uint row, uint column) {
  return subgroup * 256u + column_block * 64u + row * 8u + column;
}

#endif
