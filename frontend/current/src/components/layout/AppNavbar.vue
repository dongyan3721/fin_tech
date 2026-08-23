<script setup>
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NMenu,
  NButton,
  NDrawer,
  NDrawerContent,
} from 'naive-ui'
import { useAppStore } from '@/stores/app'

const store = useAppStore()
const route = useRoute()
const router = useRouter()
const drawerOpen = ref(false)

const menuOptions = [
  { label: '总览', key: '/dashboard' },
  { label: '供应链图谱', key: '/graph' },
  { label: '风险评测', key: '/risk' },
  { label: '企业分析', key: '/company' },
  { label: '方案对比', key: '/compare' },
]

function onMenuSelect(key) {
  drawerOpen.value = false
  if (key !== route.path) router.push(key)
}
</script>

<template>
  <header class="navbar" :class="{ 'navbar-light': !store.isDark }">
    <div class="navbar-inner">
      <router-link to="/" class="brand">
        <span class="brand-mark">◆</span>
        <span class="brand-text">供应链风险智能评估</span>
      </router-link>

      <!-- 桌面端水平菜单 -->
      <n-menu
        class="desktop-menu"
        mode="horizontal"
        :value="route.path"
        :options="menuOptions"
        responsive
        @update:value="onMenuSelect"
      />

      <div class="actions">
        <n-button quaternary circle @click="store.toggleTheme()" :title="'切换主题'">
          {{ store.isDark ? '🌙' : '☀️' }}
        </n-button>
        <n-button class="mobile-toggle" quaternary @click="drawerOpen = true">☰</n-button>
      </div>
    </div>

    <!-- 移动端抽屉菜单 -->
    <n-drawer v-model:show="drawerOpen" placement="right" :width="220">
      <n-drawer-content title="导航" closable>
        <n-menu
          :value="route.path"
          :options="menuOptions"
          :default-expand-all="true"
          @update:value="onMenuSelect"
        />
      </n-drawer-content>
    </n-drawer>
  </header>
</template>

<style scoped>
.navbar {
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  background: rgba(18, 22, 33, 0.72);
  border-bottom: 1px solid rgba(128, 138, 157, 0.16);
}

.navbar-light {
  background: rgba(255, 255, 255, 0.78);
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.navbar-inner {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 clamp(12px, 3vw, 32px);
  height: 60px;
  display: flex;
  align-items: center;
  gap: 24px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: inherit;
  white-space: nowrap;
}

.brand-mark {
  color: #36ad6a;
  font-size: 20px;
  text-shadow: 0 0 12px rgba(54, 173, 106, 0.55);
}

.brand-text {
  font-size: 17px;
  font-weight: 700;
  letter-spacing: 1px;
}

.desktop-menu {
  flex: 1;
}

.actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
}

.mobile-toggle {
  display: none;
}

@media (max-width: 820px) {
  .desktop-menu {
    display: none;
  }
  .mobile-toggle {
    display: inline-flex;
  }
}
</style>
