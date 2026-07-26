<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-header">
        <div class="login-logo">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
          </svg>
        </div>
        <h1>AI 智能助手</h1>
        <p>请登录后继续</p>
      </div>

      <div class="login-form">
        <div class="field">
          <label>用户名</label>
          <input
            v-model="username"
            type="text"
            placeholder="请输入用户名"
            @keydown.enter="handleLogin"
            :disabled="loading"
          />
        </div>
        <div class="field">
          <label>密码</label>
          <input
            v-model="password"
            type="password"
            placeholder="请输入密码"
            @keydown.enter="handleLogin"
            :disabled="loading"
          />
        </div>

        <p v-if="error" class="error-msg">{{ error }}</p>

        <button class="login-btn" :disabled="!username || !password || loading" @click="handleLogin">
          {{ loading ? '登录中...' : '登 录' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const emit = defineEmits(['login-success'])

const username = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  if (!username.value || !password.value || loading.value) return
  loading.value = true
  error.value = ''

  try {
    const res = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username.value, password: password.value }),
    })
    const data = await res.json()
    if (data.code === 0) {
      sessionStorage.setItem('token', data.data.token)
      sessionStorage.setItem('username', data.data.username)
      emit('login-success', data.data)
    } else {
      error.value = data.message || '登录失败'
    }
  } catch (e) {
    error.value = '请求失败：' + e.message
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.login-card {
  background: #fff;
  border-radius: 16px;
  padding: 40px;
  width: 380px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.15);
}
.login-header {
  text-align: center;
  margin-bottom: 32px;
}
.login-logo {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: #fff;
  margin-bottom: 16px;
}
.login-header h1 {
  font-size: 22px;
  color: #1a1a1a;
  margin-bottom: 6px;
}
.login-header p {
  font-size: 14px;
  color: #999;
}
.field {
  margin-bottom: 20px;
}
.field label {
  display: block;
  font-size: 13px;
  color: #666;
  margin-bottom: 6px;
  font-weight: 500;
}
.field input {
  width: 100%;
  padding: 12px 14px;
  border: 1px solid #e8e8e8;
  border-radius: 10px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
  box-sizing: border-box;
}
.field input:focus {
  border-color: #667eea;
}
.error-msg {
  color: #ff4d4f;
  font-size: 13px;
  margin-bottom: 16px;
}
.login-btn {
  width: 100%;
  padding: 12px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: #fff;
  border: none;
  border-radius: 10px;
  font-size: 15px;
  cursor: pointer;
  transition: opacity 0.2s;
}
.login-btn:hover:not(:disabled) { opacity: 0.9; }
.login-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
