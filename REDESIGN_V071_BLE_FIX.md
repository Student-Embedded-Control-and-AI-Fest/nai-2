# NoodleAI Web v0.7.1 BLE scope/cache fix

Fixes `noodleBLE is not defined` seen after the visual redesign.

Changes:
- exposes `NoodleAIBLE` explicitly as `window.NoodleAIBLE`
- accesses the BLE singleton explicitly as `window.noodleBLE`
- adds a clear startup guard if `js/ble.js` did not load
- adds `?v=071` cache-busting to local CSS/JS assets so GitHub Pages/browser
  cannot mix a new `index.html` with stale JavaScript files

The BLE protocol and application behavior are otherwise unchanged.
