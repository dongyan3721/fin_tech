<script setup>
import { NCard } from 'naive-ui'

defineProps({
  label: { type: String, required: true },
  value: { type: String, required: true },
  sub: { type: String, default: '' },
  accent: { type: String, default: '#36ad6a' },
})
</script>

<template>
  <n-card size="small" class="metric-card">
    <div class="label">{{ label }}</div>
    <div class="value" :style="{ color: accent }">{{ value }}</div>
    <div v-if="sub" class="sub">{{ sub }}</div>
  </n-card>
</template>

<style scoped>
.metric-card {
  border-radius: 12px;
}

.label {
  font-size: 13px;
  opacity: 0.62;
  letter-spacing: 0.5px;
}

.value {
  margin-top: 8px;
  font-size: clamp(24px, 3vw, 34px);
  font-weight: 700;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}

.sub {
  margin-top: 6px;
  font-size: 12px;
  opacity: 0.55;
}
</style>
