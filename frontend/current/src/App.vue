<script setup>
import { computed } from 'vue'
import {
  NConfigProvider,
  NMessageProvider,
  NDialogProvider,
  NGlobalStyle,
  darkTheme,
  zhCN,
  dateZhCN,
} from 'naive-ui'
import { useAppStore } from '@/stores/app'
import AppNavbar from '@/components/layout/AppNavbar.vue'

const store = useAppStore()
const naiveTheme = computed(() => (store.isDark ? darkTheme : null))
</script>

<template>
  <n-config-provider
    :theme="naiveTheme"
    :locale="zhCN"
    :date-locale="dateZhCN"
    :theme-overrides="{
      common: {
        borderRadius: '10px',
        primaryColor: '#36ad6a',
        primaryColorHover: '#5acf8a',
      },
    }"
  >
    <n-message-provider>
      <n-dialog-provider>
        <n-global-style />
        <div class="app-shell">
          <AppNavbar />
          <main class="app-main">
            <router-view />
          </main>
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-main {
  flex: 1;
  width: 100%;
  max-width: 1440px;
  margin: 0 auto;
  padding: 24px clamp(12px, 3vw, 32px) 48px;
}
</style>
