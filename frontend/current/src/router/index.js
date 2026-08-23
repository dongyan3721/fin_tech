import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/dashboard' },
  {
    path: '/dashboard',
    name: 'dashboard',
    component: () => import('@/views/DashboardView.vue'),
    meta: { title: '总览' },
  },
  {
    path: '/graph',
    name: 'graph',
    component: () => import('@/views/GraphView.vue'),
    meta: { title: '供应链图谱' },
  },
  {
    path: '/risk',
    name: 'risk',
    component: () => import('@/views/RiskEvalView.vue'),
    meta: { title: '风险评测' },
  },
  {
    path: '/company',
    name: 'company',
    component: () => import('@/views/CompanyView.vue'),
    meta: { title: '企业分析' },
  },
  {
    path: '/compare',
    name: 'compare',
    component: () => import('@/views/CompareView.vue'),
    meta: { title: '方案对比' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.afterEach((to) => {
  document.title = to.meta.title
    ? `${to.meta.title} · 供应链风险智能评估`
    : '供应链风险智能评估'
})

export default router
