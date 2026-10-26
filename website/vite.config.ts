import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const processEnv = (globalThis as typeof globalThis & {
  process?: { env?: Record<string, string | undefined> }
}).process?.env

export default defineConfig({
  plugins: [react()],
  base: processEnv?.GITHUB_ACTIONS ? '/divy/' : '/',
})
