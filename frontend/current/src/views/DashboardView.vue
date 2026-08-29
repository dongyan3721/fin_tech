<script setup>
import { ref, computed, watch, onMounted, h } from 'vue'
import { NGrid, NGi, NDataTable, NEmpty, NSpin } from 'naive-ui'
import { api } from '@/api/client'
import { useAppStore } from '@/stores/app'
import PageHeader from '@/components/layout/PageHeader.vue'
import MetricCard from '@/components/data/MetricCard.vue'
import RatingTag from '@/components/data/RatingTag.vue'
import RunSelecter from '@/components/data/RunSelecter.vue'
import ChartCard from '@/components/charts/ChartCard.vue'
import RatingDonut from '@/components/charts/RatingDonut.vue'
import ProbHistogram from '@/components/charts/ProbHistogram.vue'
import TrainingCurve from '@/components/charts/TrainingCurve.vue'
import ScatterActualPred from '@/components/charts/ScatterActualPred.vue'
import { fmtProb } from '@/utils/rating'

const store = useAppStore()

const INF_YEAR = new Date().getFullYear()

const meta = ref(null)
const metrics = ref(null)
const evalLog = ref([])
const testItems = ref([])
const futureItems = ref([])
const infMeta = ref({ year: INF_YEAR, n_companies: 0 })
const loading = ref(false)
const evalEmpty = ref(false)

const probLabel = computed(() => {
  const s = metrics.value?.label_scheme
  if (s === 'market') return '平均预测市场风险'
  if (s === 'mix') return '平均综合风险评分'
  return '平均预测违约概率'
})

const probShort = computed(() => {
  const s = metrics.value?.label_scheme
  if (s === 'market') return '市场风险'
  if (s === 'mix') return '综合评分'
  return '违约概率'
})

async function loadMeta() {
  meta.value = await api.meta()
  store.setRuns(meta.value.runs, meta.value.latest_run)
}

async function loadRun() {
  const run = store.selectedRun
  if (!run) return
  loading.value = true
  try {
    const [m, t] = await Promise.all([
      api.modelMetrics(run),
      api.predictionsTest(run),
    ])
    metrics.value = m
    testItems.value = t.items

    // 实时推理失败不应拖垮其它图表（top=1000 覆盖全部参与公司）
    try {
      const f = await api.inference(run, INF_YEAR, 1000)
      futureItems.value = f.items
      infMeta.value = { year: f.year, n_companies: f.n_companies }
    } catch {
      futureItems.value = []
      infMeta.value = { year: INF_YEAR, n_companies: 0 }
    }

    evalEmpty.value = false
    try {
      const e = await api.modelEvalLog(run)
      evalLog.value = e.points || []
      evalEmpty.value = evalLog.value.length === 0
    } catch {
      evalLog.value = []
      evalEmpty.value = true
    }
  } finally {
    loading.value = false
  }
}

watch(() => store.selectedRun, loadRun)

onMounted(async () => {
  await loadMeta()
  await loadRun()
})

const topFuture = computed(() => futureItems.value.slice(0, 10))

const avgProb = computed(() => {
  const arr = futureItems.value.length
    ? futureItems.value
    : testItems.value.map((i) => ({ predicted_probability: i.predicted_probability }))
  const probs = arr.map((i) => i.predicted_probability).filter((p) => p != null)
  if (!probs.length) return '—'
  return fmtProb(probs.reduce((a, b) => a + b, 0) / probs.length)
})

const topColumns = [
  { title: '排名', key: 'rank', width: 56 },
  { title: '股票代码', key: 'symbol', width: 100 },
  {
    title: probShort,
    key: 'predicted_probability',
    render: (row) => fmtProb(row.predicted_probability, 3),
  },
  {
    title: '评级',
    key: 'risk_rating',
    render: (row) => h(RatingTag, { rating: row.risk_rating }),
  },
]

function topRows() {
  return topFuture.value.map((it, i) => ({ ...it, rank: i + 1 }))
}
</script>

<template>
  <div>
    <PageHeader title="总览" subtitle="数据规模、模型指标与高风险榜，一屏总览">
      <template #extra>
        <RunSelecter v-if="meta" :runs="meta.runs" />
      </template>
    </PageHeader>

    <n-spin :show="loading">
      <!-- KPI 卡片 -->
      <n-grid cols="2 s:2 m:3 l:5" :x-gap="16" :y-gap="16" responsive="screen">
        <n-gi>
          <MetricCard label="覆盖公司" :value="meta ? String(meta.n_companies) : '—'"
            sub="拥有财务特征的公司" accent="#36ad6a" />
        </n-gi>
        <n-gi>
          <MetricCard label="供应链关系" :value="meta ? String(meta.n_edges) : '—'"
            :sub="meta ? `${meta.graph_years[0]}–${meta.graph_years[meta.graph_years.length - 1]} 年` : ''"
            accent="#4c8dff" />
        </n-gi>
        <n-gi>
          <MetricCard :label="probLabel" :value="avgProb"
            :sub="futureItems.length ? `${INF_YEAR} 年预测 · ${infMeta.n_companies} 家` : '—'" accent="#f0a020" />
        </n-gi>
        <n-gi>
          <MetricCard label="R²(prob)" :value="metrics ? metrics.r2.toFixed(4) : '—'"
            :sub="metrics && metrics.spearman != null ? `Spearman ${metrics.spearman.toFixed(3)}` : ''"
            accent="#9a6fe0" />
        </n-gi>
        <n-gi v-if="metrics && metrics.auc != null">
          <MetricCard label="AUC（判别力）" :value="metrics.auc.toFixed(4)"
            :sub="`KS ${metrics.ks.toFixed(3)} · 违约样本 ${metrics.n_default ?? 0}`" accent="#e05a5a" />
        </n-gi>
      </n-grid>

      <!-- 图表区 -->
      <n-grid cols="1 m:2" :x-gap="16" :y-gap="16" responsive="screen" style="margin-top: 16px">
        <n-gi>
          <ChartCard title="测试集评级分布" :loading="loading" :empty="!testItems.length">
            <RatingDonut :items="testItems" />
          </ChartCard>
        </n-gi>
        <n-gi>
          <ChartCard title="预测违约概率分布" :loading="loading" :empty="!testItems.length">
            <ProbHistogram :items="testItems" />
          </ChartCard>
        </n-gi>
        <n-gi>
          <ChartCard title="训练过程（R² / MSE）" :loading="loading" :empty="evalEmpty"
            empty-text="该 run 未记录评估日志（eval_every=0），可切换其他 run 查看">
            <TrainingCurve :points="evalLog" />
          </ChartCard>
        </n-gi>
        <n-gi>
          <ChartCard title="实际 vs 预测（测试集）" :loading="loading" :empty="!testItems.length">
            <ScatterActualPred :items="testItems" />
          </ChartCard>
        </n-gi>
      </n-grid>

      <!-- 高风险榜 -->
      <ChartCard :loading="loading" :empty="!topFuture.length"
        empty-text="该模型暂无推理结果" style="margin-top: 16px">
        <template #header>
          <div class="top-header">
            <span class="top-title">高风险榜 Top 10（{{ INF_YEAR }} 年预测）</span>
            <span class="top-note">基于 {{ INF_YEAR - 3 }}–{{ INF_YEAR - 1 }} 年特征实时推理</span>
          </div>
        </template>
        <n-data-table
          :columns="topColumns"
          :data="topRows()"
          :bordered="false"
          size="small"
          :max-height="420"
        />
      </ChartCard>
    </n-spin>
  </div>
</template>

<style scoped>
.top-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  flex-wrap: wrap;
}

.top-title {
  font-weight: 600;
  font-size: 15px;
}

.top-note {
  font-size: 12px;
  opacity: 0.55;
}
</style>
