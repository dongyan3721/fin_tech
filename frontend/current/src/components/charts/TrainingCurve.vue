<script setup>
import { computed } from 'vue'
import VChart from '@/plugins/echarts'
import { chartPalette } from '@/utils/chartTheme'
import { useAppStore } from '@/stores/app'

const props = defineProps({
  points: { type: Array, default: () => [] },
})

const store = useAppStore()

const option = computed(() => {
  const pal = chartPalette(store.isDark)
  const epochs = props.points.map((p) => p.epoch)
  const r2 = props.points.map((p) => p.r2)
  const mse = props.points.map((p) => p.mse)
  return {
    tooltip: {
      trigger: 'axis',
      backgroundColor: pal.tooltipBg,
      borderColor: pal.tooltipBorder,
      textStyle: { color: pal.tooltipText },
    },
    legend: { data: ['R²(prob)', 'MSE'], textStyle: { color: pal.text }, top: 0 },
    grid: { left: 8, right: 8, top: 32, bottom: 8, containLabel: true },
    xAxis: {
      type: 'category',
      data: epochs,
      axisLabel: { color: pal.subtext },
      axisLine: { lineStyle: { color: pal.splitLine } },
    },
    yAxis: [
      {
        type: 'value',
        name: 'R²',
        splitLine: { lineStyle: { color: pal.splitLine } },
        axisLabel: { color: pal.subtext },
        nameTextStyle: { color: pal.subtext },
      },
      {
        type: 'value',
        name: 'MSE',
        splitLine: { show: false },
        axisLabel: { color: pal.subtext },
        nameTextStyle: { color: pal.subtext },
      },
    ],
    series: [
      {
        name: 'R²(prob)',
        type: 'line',
        smooth: true,
        data: r2,
        itemStyle: { color: '#36ad6a' },
        lineStyle: { width: 2 },
      },
      {
        name: 'MSE',
        type: 'line',
        smooth: true,
        yAxisIndex: 1,
        data: mse,
        itemStyle: { color: '#4c8dff' },
        lineStyle: { width: 2 },
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
