<script setup>
import { ref, computed, watch, onMounted, h } from 'vue'
import { NGrid, NGi, NInputNumber, NButton, NDataTable, useMessage } from 'naive-ui'
import { api } from '@/api/client'
import { useAppStore } from '@/stores/app'
import PageHeader from '@/components/layout/PageHeader.vue'
import MetricCard from '@/components/data/MetricCard.vue'
import RatingTag from '@/components/data/RatingTag.vue'
import RunSelecter from '@/components/data/RunSelecter.vue'
import ChartCard from '@/components/charts/ChartCard.vue'
import ScatterActualPred from '@/components/charts/ScatterActualPred.vue'
import RatingDonut from '@/components/charts/RatingDonut.vue'
import { fmtProb } from '@/utils/rating'

const store = useAppStore()
const message = useMessage()

const meta = ref(null)
const metrics = ref(null)
const testItems = ref([])
const loading = ref(false)

// —— 实时推理 ——
const INF_YEAR = new Date().getFullYear()
const infYear = ref(INF_YEAR)
const infItems = ref([])
const infMeta = ref({ n_companies: 0, year: INF_YEAR })
const infLoading = ref(false)

const probShort = computed(() => {
  const s = metrics.value?.label_scheme
  if (s === 'market') return '市场风险'
  if (s === 'mix') return '综合评分'
  return '违约概率'
})

const infColumns = computed(() => [
  { title: '排名', key: 'rank', width: 64 },
  { title: '股票代码', key: 'symbol', width: 120 },
  {
    title: `预测${probShort.value}`,
    key: 'predicted_probability',
    render: (r) => fmtProb(r.predicted_probability, 3),
  },
  {
    title: '评级/分档',
    key: 'risk_rating',
    render: (r) => h(RatingTag, { rating: r.risk_rating }),
  },
])

async function runInference() {
  if (!store.selectedRun) return
  infLoading.value = true
  try {
    const r = await api.inference(store.selectedRun, infYear.value, 50)
    infItems.value = r.items
    infMeta.value = { n_companies: r.n_companies, year: r.year }
    if (!r.items.length) {
      message.warning(`${r.year} 年无可推理公司（特征不足）`)
    }
  } catch (e) {
    message.error(`推理失败：${e.message}`)
  } finally {
    infLoading.value = false
  }
}

function exportCsv() {
  const head = ['rank', 'symbol', 'predicted_probability', 'risk_rating']
  const lines = [head.join(',')]
  for (const r of infItems.value) {
    lines.push([r.rank, r.symbol, r.predicted_probability, r.risk_rating ?? ''].join(','))
  }
  const blob = new Blob(['\uFEFF' + lines.join('\n')], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `Top50_${store.selectedRun}_${infYear.value}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
  message.success(`已导出 ${infItems.value.length} 行`)
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
    await runInference()
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
    <PageHeader title="风险评测" subtitle="模型指标、实际 vs 预测，以及按年份的实时风险推理">
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

    <!-- 实时推理 Top50 -->
    <ChartCard :loading="infLoading" :empty="!infItems.length"
      empty-text="输入年份后点击「计算」" style="margin-top: 16px">
      <template #header>
        <div class="inf-header">
          <span class="title">实时风险推理 Top 50（{{ infMeta.year }} 年预测）</span>
          <div class="filters">
            <n-input-number v-model:value="infYear" :min="2004" :max="INF_YEAR" size="small"
              style="width: 120px" @keyup.enter="runInference" />
            <n-button size="small" type="primary" :loading="infLoading" @click="runInference">计算</n-button>
            <n-button size="small" ghost :disabled="!infItems.length" @click="exportCsv">导出 CSV</n-button>
          </div>
        </div>
      </template>
      <n-data-table
        :columns="infColumns"
        :data="infItems"
        :bordered="false"
        size="small"
        :max-height="560"
      />
      <div class="inf-meta">
        参与公司 {{ infMeta.n_companies }} 家 · 基于 {{ infMeta.year - 3 }}–{{ infMeta.year - 1 }} 年特征 ·
        模型 {{ store.selectedRun }}
      </div>
    </ChartCard>
  </div>
</template>

<style scoped>
.inf-header {
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

.inf-meta {
  margin-top: 10px;
  font-size: 12px;
  opacity: 0.55;
}
</style>
