<script setup>
import { RATING_ORDER, RATING_COLORS } from '@/utils/rating'
</script>

<template>
  <div class="legend">
    <div class="legend-row">
      <span class="legend-title">评级</span>
      <span v-for="r in RATING_ORDER" :key="r" class="legend-item">
        <i class="dot" :style="{ background: RATING_COLORS[r] }" />
        {{ r }}
      </span>
    </div>
    <div class="legend-row">
      <span class="legend-title">关系</span>
      <span class="legend-item">
        <i class="edge-arrow supply" />供应
      </span>
      <span class="legend-item">
        <i class="edge-arrow sale" />销售
      </span>
    </div>
  </div>
</template>

<style scoped>
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
  align-items: center;
  font-size: 12px;
}

.legend-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.legend-title {
  opacity: 0.6;
  margin-right: 2px;
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}

.edge-arrow {
  display: inline-block;
  width: 16px;
  height: 0;
  border-top: 2px solid;
  position: relative;
}

.edge-arrow::after {
  content: '';
  position: absolute;
  right: -1px;
  top: -3px;
  border-left: 4px solid currentColor;
  border-top: 3px solid transparent;
  border-bottom: 3px solid transparent;
}

.edge-arrow.supply {
  color: #4c8dff;
}

.edge-arrow.sale {
  color: #36ad6a;
}
</style>
