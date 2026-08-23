<script setup>
import { computed } from 'vue'
import VChart from '@/plugins/echarts'
import { chartPalette } from '@/utils/chartTheme'
import { useAppStore } from '@/stores/app'

const props = defineProps({
  items: { type: Array, default: () => [] },
})

const store = useAppStore()

const option = computed(() => {
  const pal = chartPalette(store.isDark)
  const probs = props.items
    .map((it) => it.predicted_probability)
    .filter((p) => p != null)

  const BINS = 20
  const max = Math.max(0.1, ...probs)
  const step = max / BINS
  const bins = new Array(BINS).fill(0)
  for (const p of probs) {
    const idx = Math.min(BINS - 1, Math.floor(p / step))
    bins[idx] += 1
  }
  const labels = bins.map((_, i) => `${(i * step * 100).toFixed(1)}~${((i + 1) * step * 100).toFixed(1)}`)
  return {
    grid: { left: 8, right: 16, top: 24, bottom: 8, containLabel: true },
    tooltip: {
      trigger: 'axis',
      backgroundColor: pal.tooltipBg,
      borderColor: pal.tooltipBorder,
      textStyle: { color: pal.tooltipText },
      axisPointer: { type: 'shadow' },
      formatter: (p) => `${p[0].name}%<br/>样本数: ${p[0].value}`,
    },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: pal.subtext, rotate: 45, fontSize: 10, interval: 3 },
      axisLine: { lineStyle: { color: pal.splitLine } },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: pal.splitLine } },
      axisLabel: { color: pal.subtext },
    },
    series: [
      {
        type: 'bar',
        data: bins,
        itemStyle: { color: '#4c8dff', borderRadius: [4, 4, 0, 0] },
      },
    ],
  }
})
</script>

<template>
  <v-chart class="chart" :option="option" autoresize />
</template>

<style scoped>
.chart {
  height: 260px;
}
</style>
