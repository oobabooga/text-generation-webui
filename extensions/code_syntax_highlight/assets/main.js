/*
 * Code Syntax Highlight params
 * We use the <code-syntax-highlight> element as a data proxy between Gradio and
 * this JS script, the following is the list of params:
 *
 * activate: if set to true, the extension will highlight code blocks, this setting
 * must be set to true for any of the other settings to work
 *
 * inline_highlight: if set to true, code blocks without the <pre> tag (inline
 * code blocks) will also be highlighted
 *
 * performance_mode: if set to true, the extension will wait some time before highlighting the
 * code on the page and use less resources
 *
 * performance_mode_timeout_time: time the extension waits after the DOM finishes updating to
 * apply the code highlight, see explanation above the function performanceHighlight()
 *
 */
const dataProxy = document.getElementById('code-syntax-highlight');
let params = JSON.parse(dataProxy.getAttribute('params'));
// Watch for changes in the params
const paramsObserver = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.attributeName === 'params') {
      // Params were changed, update local values to reflect changes
      params = JSON.parse(dataProxy.getAttribute('params'));
    }
  });
});
paramsObserver.observe(dataProxy, {
  attributes: true,
  attributeFilter: ['params'],
  childList: false,
  subtree: false,
});

/*
 * Update highlight.js CSS theme based on current Gradio theme
 *
 * Both themes are present in the page as separate styles with
 * the media attribute set to 'not all' to keep them disabled
 *
 * We only enable one theme by setting the media attribute to 'all'
 */
function updateTheme({ theme = 'light' } = {}) {
  // Enable specified theme
  document.getElementById(`hljs-theme-${theme}`).setAttribute('media', 'all');
  // Disable opposite theme
  const themeToDisable = theme === 'light' ? 'dark' : 'light';
  document.getElementById(`hljs-theme-${themeToDisable}`).setAttribute('media', 'not all');
}

// Apply highlight.js theme when DOM loads for the first time
const gradioAppContainer = document.querySelectorAll('[class^=\'gradio\'].app, [class*=\'gradio\'].app')[0];
updateTheme({ theme: gradioAppContainer.classList.contains('dark') ? 'dark' : 'light' });

// Watch for changes in the Gradio theme and change the highlight.js theme accordingly
const themeObserver = new MutationObserver((mutations) => {
  mutations.forEach((mutation) => {
    if (mutation.attributeName === 'class') {
      // Class change was detected, reapply theme based on new class
      updateTheme({ theme: mutation.target.classList.contains('dark') ? 'dark' : 'light' });
    }
  });
});
themeObserver.observe(gradioAppContainer, {
  attributes: true,
  attributeFilter: ['class'],
  childList: false,
  subtree: false,
});

// Highlight code blocks with highlight.js
function highlightCode({ inlineHighlight = params.inline_highlight, codeElement = null } = {}) {
  // Stop if code syntax highlighting is disabled in the params
  if (params.activate === false) return;
  // Configure highlight.js to also highlight inline code blocks if specified
  const cssSelector = inlineHighlight === true ? 'code' : 'pre code';
  hljs.configure({
    cssSelector,
    ignoreUnescapedHTML: true,
    throwUnescapedHTML: false,
  });
  // Highlight just the provided code element or every code element in the DOM
  if (!codeElement) hljs.highlightAll();
  else hljs.highlightElement(codeElement);
}

// Apply highlight to all code blocks present in the DOM when it first loads
if (params.activate === true) highlightCode();

/*
 * This is the logic behind how we apply the highlight in performance mode,
 * so that we don't call highlightCode() for each token during text generation:
 *
 * (1) DOM update is detected (text is being generated or finished generating)
 * (2) Are there any code blocks in the DOM?
 *  -> If no, stop
 *  -> If yes, continue to (3)
 * (3) We wait the time specified in 'params.performance_mode_timeout_time' (in milliseconds)
 * (4) Did the DOM update again while waiting?
 *  -> If yes, go back to (3)
 *  -> If no, continue to (5)
 * (5) We highlight all code blocks present on the page
 *
 * We have to highlight all code blocks again every time the DOM finishes
 * updating, because the text generation overrides the classes set by highlight.js
 */
let highlightTimeout;
function performanceHighlight() {
  clearTimeout(highlightTimeout);
  highlightTimeout = setTimeout(() => {
    highlightCode();
  }, params.performance_mode_timeout_time);
}

// Watch for changes in the DOM body with arrive.js to highlight new code blocks as they appear
document.body.arrive('CODE', (codeElement) => {
  // Stop if code syntax highlighting is disabled in the params
  if (params.activate === false) return;
  // Check if we need to highlight full code blocks and inline ones, or just full code blocks
  if (params.inline_highlight === false && codeElement.parentElement.nodeName !== 'PRE') return;
  // Highlight based on performance mode
  if (params.performance_mode === true) performanceHighlight();
  else highlightCode({ codeElement });
});
