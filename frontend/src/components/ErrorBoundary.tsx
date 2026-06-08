import { Component, ErrorInfo, ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  info: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null, info: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, info: null }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary] Caught:', error, info)
    this.setState({ info })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 24, fontFamily: 'monospace', background: '#fff1f2', minHeight: '100vh' }}>
          <h2 style={{ color: '#dc2626', marginBottom: 8 }}>Runtime Error — React crash</h2>
          <p style={{ color: '#7f1d1d', marginBottom: 16, fontSize: 14 }}>
            Buka browser console (F12) untuk detail lengkap.
          </p>
          <pre style={{ background: '#fee2e2', padding: 16, borderRadius: 8, fontSize: 12, overflow: 'auto', whiteSpace: 'pre-wrap', color: '#991b1b' }}>
            {this.state.error?.toString()}
            {'\n\n'}
            {this.state.info?.componentStack}
          </pre>
          <button
            onClick={() => this.setState({ hasError: false, error: null, info: null })}
            style={{ marginTop: 16, padding: '8px 16px', background: '#dc2626', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer' }}
          >
            Try Again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
