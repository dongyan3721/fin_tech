<script setup>
import { computed } from 'vue'
import VChart from '@/plugins/echarts'
import { chartPalette } from '@/utils/chartTheme'
import { useAppStore } from '@/stores/app'

const props = defineProps({
  categories: { type: Array, default: () => [] },
  series: { type: Array, default: () => [] }, // [{ name, data, color }]
  yName: { type: String, default: '' },
})

const store = useAppStore()

const option = computed(() => {
  const pal = chartPalette(store.isDark)
  return {
    tooltip: {
      trigger: 'axis',
      backgroundColor: pal.tooltipBg,
      borderColor: pal.tooltipBorder,
      textStyle: { color: pal.tooltipText },
    },
    legend: { data: props.series.map((s) => s.name), textStyle: { color: pal.text }, top: 0 },
    grid: { left: 8, right: 16, top: 32, bottom: 8, containLabel: true },
    xAxis: {
      type: 'category',
      data: props.categories,
      axisLabel: { color: pal.subtext },
      axisLine: { lineStyle: { color: pal.splitLine } },
    },
    yAxis: {
      type: 'value',
      name: props.yName,
      nameTextStyle: { color: pal.subtext },
      splitLine: { lineStyle: { color: pal.splitLine } },
      axisLabel: { color: pal.subtext },
    },
    series: props.series.map((s) => ({
      name: s.name,
      type: 'line',
      smooth: true,
      data: s.data,
      connectNulls: false,
      symbolSize: 6,
      itemStyle: { color: s.color },
      lineStyle: { width: 2 },
    })),
  }
})
</script>

<template>
  <v-chart class="chart" :option="option" autoresize />
</template>

<style scoped>
.chart {
  height: 280px;
}
</style>
