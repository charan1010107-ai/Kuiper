import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Ignore third-party browser extension errors (e.g. MetaMask inpage.js)
if (typeof window !== "undefined") {
  const isExtensionError = (e) => {
    const msg = e?.message || e?.reason?.message || String(e || "");
    const src = e?.filename || e?.reason?.stack || "";
    return (
      msg.includes("MetaMask") ||
      src.includes("chrome-extension://") ||
      src.includes("moz-extension://") ||
      src.includes("inpage.js")
    );
  };

  window.addEventListener("error", (event) => {
    if (isExtensionError(event)) {
      event.stopImmediatePropagation();
      event.preventDefault();
    }
  }, true);

  window.addEventListener("unhandledrejection", (event) => {
    if (isExtensionError(event)) {
      event.stopImmediatePropagation();
      event.preventDefault();
    }
  }, true);
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
