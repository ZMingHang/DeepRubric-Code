(() => {
  const navigation = document.querySelector('.site-nav');
  const menuButton = document.querySelector('.menu-toggle');
  const menuIcon = menuButton.querySelector('.icon');

  function setMenuOpen(open) {
    navigation.classList.toggle('is-open', open);
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    menuButton.title = open ? 'Close navigation' : 'Open navigation';
    menuIcon.className = `icon ${open ? 'icon-x' : 'icon-menu'}`;
  }

  menuButton.hidden = false;
  document.documentElement.classList.add('js');
  menuButton.addEventListener('click', () => setMenuOpen(menuButton.getAttribute('aria-expanded') !== 'true'));
  navigation.addEventListener('click', event => {
    if (event.target.closest('a')) setMenuOpen(false);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && menuButton.getAttribute('aria-expanded') === 'true') {
      setMenuOpen(false);
      menuButton.focus();
    }
  });
  window.matchMedia('(min-width: 781px)').addEventListener('change', () => setMenuOpen(false));

  const navigationLinks = [...navigation.querySelectorAll('a')];
  const sections = navigationLinks.map(link => document.querySelector(link.hash));
  let navigationFrame;
  function updateNavigation() {
    let active = '';
    sections.forEach(section => {
      if (section.getBoundingClientRect().top <= innerHeight * 0.3) active = `#${section.id}`;
    });
    if (scrollY + innerHeight >= document.documentElement.scrollHeight - 2) active = '#bibtex';
    navigationLinks.forEach(link => {
      if (link.hash === active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    navigationFrame = null;
  }
  function scheduleNavigation() {
    if (!navigationFrame) navigationFrame = requestAnimationFrame(updateNavigation);
  }
  window.addEventListener('scroll', scheduleNavigation, { passive: true });
  window.addEventListener('resize', scheduleNavigation);
  updateNavigation();

  const copyButton = document.querySelector('.copy-button');
  const copyStatus = document.querySelector('.copy-status');
  const citation = document.querySelector('#citation-text');
  let feedbackTimer;
  copyButton.hidden = false;

  function legacyCopy(text) {
    const selection = document.getSelection();
    const savedRanges = Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index).cloneRange());
    const input = document.createElement('textarea');
    input.value = text;
    input.setAttribute('readonly', '');
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.append(input);
    input.select();
    let copied = false;
    try {
      copied = document.execCommand('copy');
    } finally {
      input.remove();
      selection.removeAllRanges();
      savedRanges.forEach(range => selection.addRange(range));
      copyButton.focus({ preventScroll: true });
    }
    if (!copied) throw new Error('Clipboard unavailable');
  }

  copyButton.addEventListener('click', async () => {
    if (copyButton.getAttribute('aria-busy') === 'true') return;
    clearTimeout(feedbackTimer);
    copyButton.setAttribute('aria-busy', 'true');
    const text = citation.textContent.trim();
    try {
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(text);
      } catch {
        legacyCopy(text);
      }
      copyStatus.textContent = 'Copied';
      copyButton.querySelector('.icon').className = 'icon icon-check';
      copyButton.setAttribute('aria-label', 'Citation copied');
      copyButton.title = 'Citation copied';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = document.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent = 'Copy unavailable';
      copyButton.querySelector('.icon').className = 'icon icon-copy';
      copyButton.setAttribute('aria-label', 'Copy BibTeX citation');
      copyButton.title = 'Copy BibTeX citation';
    }
    copyButton.setAttribute('aria-busy', 'false');
    feedbackTimer = setTimeout(() => {
      copyStatus.textContent = '';
      copyButton.querySelector('.icon').className = 'icon icon-copy';
      copyButton.setAttribute('aria-label', 'Copy BibTeX citation');
      copyButton.title = 'Copy BibTeX citation';
    }, 2500);
  });

  const dialog = document.querySelector('.figure-dialog');
  if (typeof dialog.showModal === 'function') {
    const dialogImage = dialog.querySelector('.dialog-image');
    const dialogCaption = dialog.querySelector('#figure-dialog-caption');
    const canvas = dialog.querySelector('.dialog-canvas');
    const resolutionButton = dialog.querySelector('.figure-resolution');
    function setFullResolution(fullResolution) {
      canvas.classList.toggle('is-zoomed', fullResolution);
      resolutionButton.setAttribute('aria-pressed', String(fullResolution));
      resolutionButton.setAttribute('aria-label', fullResolution ? 'Fit figure' : 'View at full resolution');
      resolutionButton.title = fullResolution ? 'Fit figure' : 'View at full resolution';
      resolutionButton.querySelector('.icon').className = `icon ${fullResolution ? 'icon-zoom-out' : 'icon-zoom-in'}`;
      canvas.scrollTop = 0;
      canvas.scrollLeft = 0;
    }
    resolutionButton.addEventListener('click', () => setFullResolution(!canvas.classList.contains('is-zoomed')));
    let opener;
    document.querySelectorAll('[data-lightbox]').forEach(link => {
      link.addEventListener('click', event => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        opener = link;
        const image = link.querySelector('img');
        dialogImage.src = link.href;
        dialogImage.alt = image.alt;
        dialogCaption.textContent = link.closest('figure').querySelector('figcaption').textContent;
        dialog.querySelector('.figure-download').href = link.href;
        setFullResolution(false);
        document.body.classList.add('dialog-open');
        dialog.showModal();
      });
    });
    dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      if (event.target === dialog) {
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
      }
    });
    dialog.addEventListener('close', () => {
      document.body.classList.remove('dialog-open');
      opener?.focus({ preventScroll: true });
    });
  }
})();
