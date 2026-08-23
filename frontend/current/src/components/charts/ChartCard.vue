<script setup>
import { NCard, NSpin, NEmpty } from 'naive-ui'

defineProps({
  title: { type: String, default: '' },
  loading: { type: Boolean, default: false },
  empty: { type: Boolean, default: false },
  emptyText: { type: String, default: '暂无数据' },
})
</script>

<template>
  <n-card size="small" class="chart-card">
    <template #header>
      <slot name="header">
        <span class="title">{{ title }}</span>
      </slot>
    </template>
    <n-spin :show="loading">
      <div class="body">
        <n-empty v-if="empty" :description="emptyText" size="small" />
        <slot v-else />
      </div>
    </n-spin>
  </n-card>
</template>

<style scoped>
.chart-card {
  border-radius: 12px;
  height: 100%;
}

.title {
  font-weight: 600;
  font-size: 15px;
}

.body {
  min-height: 260px;
}
</style>
