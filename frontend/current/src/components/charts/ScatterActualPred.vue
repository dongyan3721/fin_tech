<script setup>
import { computed } from 'vue'
import VChart from '@/plugins/echarts'
import { RATING_ORDER, RATING_COLORS } from '@/utils/rating'
import { chartPalette } from '@/utils/chartTheme'
import { useAppStore } from '@/stores/app'

const props = defineProps({
  items: { type: Array, default: () => [] },
})

const store = useAppStore()

const option = computed(() => {
  const pal = chartPalette(store.isDark)
  const groups = {}
  for (const it of props.items) {
    const r = it.actual_rating || '未知'
    if (!groups[r]) groups[r] = []
    groups[r].push([it.actual_probability, it.predicted_probability])
  }
  const series = RATING_ORDER.filter((r) => groups[r]).map((r) => ({
    name: r,
    type: 'scatter',
    data: groups[r],
    symbolSize: 7,
    itemStyle: { color: RATING_COLORS[r], opacity: 0.75 },
  }))
  return {
    tooltip: {
      trigger: 'item',
      backgroundColor: pal.tooltipBg,
      borderColor: pal.tooltipBorder,
      textStyle: { color: pal.tooltipText },
      formatter: (p) => `实际: ${(p.value[0] * 100).toFixed(2)}%<br/>预测: ${(p.value[1] * 100).toFixed(2)}%`,
    },
    legend: { data: RATING_ORDER.filter((r) => groups[r]), textStyle: { color: pal.text }, top: 0 },
    grid: { left: 8, right: 20, top: 32, bottom: 8, containLabel: true },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: '实际概率',
      nameTextStyle: { color: pal.subtext },
      splitLine: { lineStyle: { color: pal.splitLine } },
      axisLabel: { color: pal.subtext, formatter: (v) => `${(v * 100).toFixed(0)}%` },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      name: '预测概率',
      nameTextStyle: { color: pal.subtext },
      splitLine: { lineStyle: { color: pal.splitLine } },
      axisLabel: { color: pal.subtext, formatter: (v) => `${(v * 100).toFixed(0)}%` },
    },
    series: [
      ...series,
      {
        type: 'line',
        data: [[0, 0], [1, 1]],
        symbol: 'none',
        lineStyle: { color: pal.text, type: 'dashed', width: 1, opacity: 0.4 },
        silent: true,
        tooltip: { show: false },
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
