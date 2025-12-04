import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/globals.css';

// Temporarily disabled MSW to allow real API requests
async function enableMocking() {
  // Only enable if PUBLIC_USE_MOCK is true or not set in dev
  if (import.meta.env.PUBLIC_USE_MOCK === 'false') {
    return;
  }

  if (process.env.NODE_ENV !== 'development') {
    return
  }
  
  const { worker } = await import('./mocks/browser')
  
  // `worker.start()` returns a Promise that resolves
  // once the Service Worker is up and ready to intercept requests.
  return worker.start()
}

const rootEl = document.getElementById('root');
if (rootEl) {
  const root = ReactDOM.createRoot(rootEl);
  
  enableMocking().then(() => {
    root.render(
      <React.StrictMode>
        <App />
      </React.StrictMode>,
    );
  })
}
