<script setup>
import { ref, computed, watch, onMounted, h } from 'vue'
import { NGrid, NGi, NSelect, NInput, NButton, NDataTable, useMessage } from 'naive-ui'
import { api } from '@/api/client'
import { useAppStore } from '@/stores/app'
import PageHeader from '@/components/layout/PageHeader.vue'
import MetricCard from '@/components/data/MetricCard.vue'
import RatingTag from '@/components/data/RatingTag.vue'
import RunSelecter from '@/components/data/RunSelecter.vue'
import ChartCard from '@/components/charts/ChartCard.vue'
import ScatterActualPred from '@/components/charts/ScatterActualPred.vue'
import RatingDonut from '@/components/charts/RatingDonut.vue'
import { RATING_ORDER, fmtProb } from '@/utils/rating'

const store = useAppStore()
const message = useMessage()

const meta = ref(null)
const metrics = ref(null)
const testItems = ref([])
const loading = ref(false)

const filterRating = ref(null)
const filterYear = ref(null)
const search = ref('')

const RATING_OPTIONS = RATING_ORDER.map((r) => ({ label: r, value: r }))

const filtered = computed(() => {
  let arr = testItems.value
  if (filterRating.value) arr = arr.filter((i) => i.actual_rating === filterRating.value)
  if (filterYear.value) arr = arr.filter((i) => i.prediction_year === filterYear.value)
  if (search.value) {
    const q = search.value.trim()
    arr = arr.filter((i) => String(i.symbol).includes(q))
  }
  return arr
})

const yearOptions = computed(() => {
  const ys = [...new Set(testItems.value.map((i) => i.prediction_year))].sort((a, b) => a - b)
  return ys.map((y) => ({ label: String(y), value: y }))
})

const columns = [
  { title: '代码', key: 'symbol', width: 96 },
  { title: '预测年', key: 'prediction_year', width: 80 },
  {
    title: '实际概率', key: 'actual_probability', width: 110,
    render: (r) => fmtProb(r.actual_probability, 3),
  },
  {
    title: '预测概率', key: 'predicted_probability', width: 110,
    render: (r) => fmtProb(r.predicted_probability, 3),
  },
  { title: '实际评级', key: 'actual_rating', width: 96, render: (r) => h(RatingTag, { rating: r.actual_rating }) },
  { title: '预测评级', key: 'predicted_rating', width: 96, render: (r) => h(RatingTag, { rating: r.predicted_rating }) },
]

function exportCsv() {
  const head = ['symbol', 'prediction_year', 'actual_probability', 'predicted_probability', 'actual_rating', 'predicted_rating']
  const lines = [head.join(',')]
  for (const r of filtered.value) {
    lines.push([r.symbol, r.prediction_year, r.actual_probability, r.predicted_probability, r.actual_rating, r.predicted_rating].join(','))
  }
  const blob = new Blob(['\uFEFF' + lines.join('\n')], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `风险评测_${store.selectedRun || 'latest'}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
  message.success(`已导出 ${filtered.value.length} 行`)
}

async function loadMeta() {
  meta.value = await api.meta()
  store.setRuns(meta.value.runs, meta.value.latest_run)
}

async function loadRun() {
  const run = store.selectedRun
  if (!run) return
  loading.value = true
  try {
    const [m, t] = await Promise.all([api.modelMetrics(run), api.predictionsTest(run)])
    metrics.value = m
    testItems.value = t.items
  } finally {
    loading.value = false
  }
}

watch(() => store.selectedRun, loadRun)

onMounted(async () => {
  await loadMeta()
  await loadRun()
})
</script>

<template>
  <div>
    <PageHeader title="风险评测" subtitle="模型指标、实际 vs 预测与 TGC/KMV 对照表">
      <template #extra>
        <RunSelecter v-if="meta" :runs="meta.runs" />
      </template>
    </PageHeader>

    <!-- 指标卡 -->
    <n-grid cols="2 s:2 m:3 l:6" :x-gap="16" :y-gap="16" responsive="screen">
      <n-gi><MetricCard label="R²(prob)" :value="metrics ? metrics.r2.toFixed(4) : '—'" accent="#36ad6a" /></n-gi>
      <n-gi><MetricCard label="R²(logit)" :value="metrics && metrics.r2_logit != null ? metrics.r2_logit.toFixed(4) : '—'" accent="#4c8dff" /></n-gi>
      <n-gi><MetricCard label="Spearman" :value="metrics && metrics.spearman != null ? metrics.spearman.toFixed(3) : '—'" accent="#9a6fe0" /></n-gi>
      <n-gi><MetricCard label="IC(Pearson)" :value="metrics && metrics.ic != null ? metrics.ic.toFixed(3) : '—'" accent="#2ec7c9" /></n-gi>
      <n-gi><MetricCard label="AUC" :value="metrics && metrics.auc != null ? metrics.auc.toFixed(4) : '—'" accent="#f0a020" /></n-gi>
      <n-gi><MetricCard label="KS" :value="metrics && metrics.ks != null ? metrics.ks.toFixed(3) : '—'" accent="#e05a5a" /></n-gi>
    </n-grid>

    <!-- 图表 -->
    <n-grid cols="1 m:2" :x-gap="16" :y-gap="16" responsive="screen" style="margin-top: 16px">
      <n-gi>
        <ChartCard title="实际 vs 预测（测试集）" :loading="loading" :empty="!testItems.length">
          <ScatterActualPred :items="testItems" />
        </ChartCard>
      </n-gi>
      <n-gi>
        <ChartCard title="评级分布（实际）" :loading="loading" :empty="!testItems.length">
          <RatingDonut :items="testItems" />
        </ChartCard>
      </n-gi>
    </n-grid>

    <!-- 对照表 -->
    <ChartCard :loading="loading" :empty="!testItems.length" style="margin-top: 16px">
      <template #header>
        <div class="table-header">
          <span class="title">TGC vs KMV 对照表</span>
          <div class="filters">
            <n-select v-model:value="filterRating" :options="RATING_OPTIONS" placeholder="评级" clearable size="small" style="width: 100px" />
            <n-select v-model:value="filterYear" :options="yearOptions" placeholder="年份" clearable size="small" style="width: 100px" />
            <n-input v-model:value="search" placeholder="代码搜索" clearable size="small" style="width: 130px" />
            <n-button size="small" type="primary" ghost @click="exportCsv">导出 CSV</n-button>
          </div>
        </div>
      </template>
      <n-data-table
        :columns="columns"
        :data="filtered"
        :bordered="false"
        size="small"
        :scroll-x="640"
        :pagination="{ pageSize: 20 }"
        :max-height="520"
      />
    </ChartCard>
  </div>
</template>

<style scoped>
.table-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.title {
  font-weight: 600;
  font-size: 15px;
}

.filters {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
</style>
