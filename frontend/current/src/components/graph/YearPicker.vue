<script setup>
import { computed } from 'vue'
import { NButton, NButtonGroup, NSlider } from 'naive-ui'

const props = defineProps({
  year: { type: Number, required: true },
  min: { type: Number, required: true },
  max: { type: Number, required: true },
})

const emit = defineEmits(['update:year'])

const setYear = (y) => emit('update:year', Math.max(props.min, Math.min(props.max, y)))

const sliderVal = computed({
  get: () => props.year,
  set: (v) => setYear(v),
})
</script>

<template>
  <div class="year-picker">
    <n-button-group>
      <n-button size="small" quaternary @click="setYear(year - 1)">‹</n-button>
      <n-button size="small" quaternary @click="setYear(year + 1)">›</n-button>
    </n-button-group>
    <n-slider
      v-model:value="sliderVal"
      :min="min"
      :max="max"
      :step="1"
      :tooltip="true"
      class="slider"
      :format-tooltip="(v) => `${v} 年`"
    />
    <span class="year-label">{{ year }} 年</span>
  </div>
</template>

<style scoped>
.year-picker {
  display: flex;
  align-items: center;
  gap: 12px;
}

.slider {
  flex: 1;
  min-width: 140px;
  max-width: 360px;
}

.year-label {
  font-size: 16px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  min-width: 56px;
  text-align: right;
}
</style>
