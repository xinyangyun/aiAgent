<template>
  <div class="chat-container">
    <!-- 头部 -->
    <div class="chat-header">
      <div class="header-avatar">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
        </svg>
      </div>
      <div class="header-info">
        <div class="header-title">AI 智能助手</div>
        <div class="header-status" :class="{ online: connected }">
          {{ connected ? '在线' : '连接中...' }}
        </div>
      </div>
    </div>

    <!-- 消息列表 -->
    <div class="messages" ref="messagesRef">
      <div
        v-for="(msg, i) in messages"
        :key="i"
        class="message-row"
        :class="msg.role"
      >
        <div class="message-bubble" v-html="msg.content"></div>
        <div class="message-time">{{ msg.time }}</div>
      </div>

      <!-- 加载中 -->
      <div v-if="loading" class="message-row assistant">
        <div class="message-bubble loading">
          <span class="dot-pulse"></span>
        </div>
      </div>
    </div>

    <!-- 输入区 -->
    <div class="input-area">
      <div class="input-wrapper">
        <input
          ref="inputRef"
          v-model="inputText"
          type="text"
          placeholder="输入消息..."
          @keydown.enter="handleSend"
          :disabled="loading"
        />
        <button class="send-btn" :disabled="!inputText.trim() || loading" @click="handleSend">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted } from 'vue'
import { sendMessage } from '../api/chat.js'

const messages = ref([
  { role: 'assistant', content: '你好！我是 AI 智能助手。你可以问我任何问题，也可以查询天气。', time: '' },
])
const inputText = ref('')
const loading = ref(false)
const connected = ref(true)

const messagesRef = ref(null)
const inputRef = ref(null)

function formatTime() {
  const now = new Date()
  return now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0')
}

async function scrollToBottom() {
  await nextTick()
  if (messagesRef.value) {
    messagesRef.value.scrollTop = messagesRef.value.scrollHeight
  }
}

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || loading.value) return

  inputText.value = ''
  messages.value.push({ role: 'user', content: escapeHtml(text), time: formatTime() })
  loading.value = true
  await scrollToBottom()

  try {
    const data = await sendMessage(text)
    messages.value.push({ role: 'assistant', content: data.response || '暂无回复', time: formatTime() })
  } catch (e) {
    messages.value.push({ role: 'assistant', content: '抱歉，请求失败：' + e.message, time: formatTime() })
  } finally {
    loading.value = false
    await scrollToBottom()
    inputRef.value?.focus()
  }
}

function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

onMounted(() => {
  inputRef.value?.focus()
})
</script>

<style>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 720px;
  margin: 0 auto;
  background: #f5f5f5;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* 头部 */
.chat-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 20px;
  background: #fff;
  border-bottom: 1px solid #e8e8e8;
  flex-shrink: 0;
}
.header-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea, #764ba2);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
}
.header-title {
  font-size: 16px;
  font-weight: 600;
  color: #1a1a1a;
}
.header-status {
  font-size: 12px;
  color: #999;
}
.header-status.online {
  color: #52c41a;
}

/* 消息列表 */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.message-row {
  display: flex;
  flex-direction: column;
  max-width: 80%;
}
.message-row.user {
  align-self: flex-end;
  align-items: flex-end;
}
.message-row.assistant {
  align-self: flex-start;
  align-items: flex-start;
}
.message-bubble {
  padding: 10px 16px;
  border-radius: 18px;
  line-height: 1.5;
  font-size: 14px;
  word-break: break-word;
  white-space: pre-wrap;
}
.message-row.user .message-bubble {
  background: #1677ff;
  color: #fff;
  border-bottom-right-radius: 4px;
}
.message-row.assistant .message-bubble {
  background: #fff;
  color: #1a1a1a;
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.message-time {
  font-size: 11px;
  color: #bbb;
  margin-top: 4px;
  padding: 0 4px;
}

/* 加载动画 */
.loading {
  padding: 14px 20px !important;
}
.dot-pulse {
  display: inline-flex;
  gap: 4px;
}
.dot-pulse::before,
.dot-pulse::after {
  content: '';
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #999;
  animation: pulse 1.4s infinite;
}
.dot-pulse::before { animation-delay: 0.2s; }
.dot-pulse::after { animation-delay: 0.4s; }
@keyframes pulse {
  0%, 80%, 100% { opacity: 0.3; }
  40% { opacity: 1; }
}

/* 输入区 */
.input-area {
  padding: 12px 20px 20px;
  background: #fff;
  border-top: 1px solid #e8e8e8;
  flex-shrink: 0;
}
.input-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #f5f5f5;
  border-radius: 24px;
  padding: 4px 4px 4px 16px;
  border: 1px solid #e8e8e8;
  transition: border-color 0.2s;
}
.input-wrapper:focus-within {
  border-color: #1677ff;
}
.input-wrapper input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  font-size: 14px;
  padding: 8px 0;
  color: #1a1a1a;
}
.input-wrapper input::placeholder {
  color: #bbb;
}
.send-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: none;
  background: #1677ff;
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s, transform 0.1s;
  flex-shrink: 0;
}
.send-btn:hover:not(:disabled) {
  background: #4096ff;
}
.send-btn:active:not(:disabled) {
  transform: scale(0.95);
}
.send-btn:disabled {
  background: #d9d9d9;
  cursor: not-allowed;
}
</style>
