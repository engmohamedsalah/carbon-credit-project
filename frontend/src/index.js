import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import App from './App';
import store from './store';
import { setupErrorHandlers } from './utils/errorUtils';
import modernTheme from './theme/modernTheme';

// Import Google Fonts for Inter and JetBrains Mono
const linkElement = document.createElement('link');
linkElement.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap';
linkElement.rel = 'stylesheet';
document.head.appendChild(linkElement);

// Add custom CSS for glassmorphism support
const styleElement = document.createElement('style');
styleElement.textContent = `
  @supports (backdrop-filter: blur(10px)) {
    .glass-effect {
      backdrop-filter: blur(20px);
    }
  }
  
  body {
    background: linear-gradient(135deg, #0A0E27 0%, #151B3B 50%, #1E2449 100%);
    background-attachment: fixed;
    min-height: 100vh;
  }
  
  /* Scrollbar styling for dark theme */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  
  ::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
  }
  
  ::-webkit-scrollbar-thumb {
    background: rgba(0, 255, 136, 0.3);
    border-radius: 4px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 255, 136, 0.5);
  }
`;
document.head.appendChild(styleElement);

// Setup global error handlers to suppress Chrome extension errors
setupErrorHandlers();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={modernTheme}>
          <CssBaseline />
          <App />
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  </React.StrictMode>
);
