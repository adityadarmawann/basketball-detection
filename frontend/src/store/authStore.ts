import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface User {
  name: string
  email: string
}

interface AuthState {
  user: User | null
  isLoggedIn: boolean
  login: (email: string, name?: string) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isLoggedIn: false,
      login: (email, name) =>
        set({
          user: { email, name: name || email.split('@')[0] },
          isLoggedIn: true,
        }),
      logout: () => set({ user: null, isLoggedIn: false }),
    }),
    { name: 'sv-auth' },
  ),
)
