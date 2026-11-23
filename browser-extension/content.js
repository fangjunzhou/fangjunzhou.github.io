(() => {
  const article = document.querySelector("article .single-content");
  const sourceData = document.querySelector("#markdown-reference-data");
  if (!article || !sourceData) return;

  const { path, records } = JSON.parse(sourceData.textContent);
  const mediaSelector = "img, video, audio";
  const blockSelector = "p, h1, h2, h3, h4, h5, h6, li, dt, dd, tr, pre, summary, .katex-display";
  const wordPattern = /[\p{L}\p{N}\p{M}]+/gu;

  // Collect DOM offsets alongside text so a selection within a wrapped
  // paragraph still points to the Markdown lines it actually touches.
  function readBlock(block) {
    let text = "";
    const pieces = [];

    function append(node, value) {
      if (!value) return;
      pieces.push({ node, from: text.length, to: text.length + value.length });
      text += value;
    }

    function visit(node) {
      if (node.nodeType === Node.TEXT_NODE) {
        append(node, node.data);
        return;
      }
      if (!(node instanceof Element) || (node !== block && node.matches(`${blockSelector}, figure, button, .anchor`))) return;

      if (node.matches(".katex")) {
        const tex = node.querySelector("annotation")?.textContent;
        if (tex) append(node, ` ${tex} `);
        return;
      }

      if (node.matches("br, td, th")) append(node, " ");
      node.childNodes.forEach(visit);
    }

    visit(block);
    const words = [...text.matchAll(wordPattern)];
    return { text, pieces, words, normalized: words.map(word => word[0]).join(" ") };
  }

  function textRange(pieces, from, to) {
    const range = document.createRange();
    const start = pieces.find(piece => piece.to > from);
    const end = pieces.find(piece => piece.to >= to);
    if (start.node instanceof Text) range.setStart(start.node, from - start.from);
    else range.setStartBefore(start.node);
    if (end.node instanceof Text) range.setEnd(end.node, to - end.from);
    else range.setEndAfter(end.node);
    return range;
  }

  function mediaURLs(media) {
    const sources = Array.from(media.querySelectorAll("source[src]"), node => node.src);
    return [media.currentSrc, media.src, ...sources].filter(Boolean);
  }

  function matchesMedia(record, urls) {
    if (!URL.canParse(record.media, document.baseURI)) return false;
    const source = new URL(record.media, document.baseURI);
    return urls.some(url => {
      const rendered = new URL(url, document.baseURI);
      if (source.origin === rendered.origin && source.pathname === rendered.pathname) return true;
      // Hugo may publish a relative asset under the site's root or bundle.
      return !/^(?:\/|[\w+.-]+:)/.test(record.media)
        && rendered.origin === location.origin
        && rendered.pathname.endsWith(new URL(record.media, location.origin).pathname);
    });
  }

  function indexArticle() {
    const lineRanges = [];
    const mediaRecords = new Map();
    let cursor = 0;

    for (const block of article.querySelectorAll(`${blockSelector}, figure, ${mediaSelector}`)) {
      const figure = block.closest("figure");
      if (figure && figure !== block) continue;

      const media = block.matches(mediaSelector) ? block : figure?.querySelector(mediaSelector);
      if (media) {
        const urls = mediaURLs(media);
        const index = records.findIndex((record, index) => index >= cursor && record.media && matchesMedia(record, urls));
        if (index !== -1) {
          const range = document.createRange();
          range.selectNode(block);
          mediaRecords.set(media, records[index]);
          lineRanges.push({ ...records[index], range });
          cursor = index + 1;
        }
        continue;
      }

      const { text, pieces, words, normalized } = readBlock(block);
      if (!words.length) continue;

      // Match whole blocks in reading order to distinguish repeated lines and
      // avoid assigning a source line to generated text that has no match.
      for (let index = cursor; index < records.length; index++) {
        let offset = 0;
        let wordIndex = 0;
        const lines = [];
        for (let next = index; next < records.length && wordIndex < words.length; next++) {
          const record = records[next];
          if (!record.text || !normalized.startsWith(record.text, offset)) break;
          offset += record.text.length;
          if (offset < normalized.length && normalized[offset] !== " ") break;
          offset++;
          const end = wordIndex + record.text.split(" ").length;
          lines.push({ record, from: wordIndex === 0 ? 0 : words[wordIndex].index, to: words[end]?.index ?? text.length });
          wordIndex = end;
        }
        if (wordIndex !== words.length) continue;

        for (const { record, from, to } of lines) {
          lineRanges.push({ ...record, range: textRange(pieces, from, to) });
        }
        cursor = index + lines.length;
        break;
      }
    }

    return { lineRanges, mediaRecords };
  }

  let contextMedia;
  let contextSelection;
  document.addEventListener("contextmenu", event => {
    const target = event.target;
    contextMedia = target instanceof Element && article.contains(target) ? target.closest(mediaSelector) : null;
    contextSelection = readSelection();
  }, true);

  function readSelection() {
    const selection = document.getSelection();
    if (!selection || selection.isCollapsed) return;
    const ranges = Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index).cloneRange());
    if (ranges.every(range => article.contains(range.startContainer) && article.contains(range.endContainer))) {
      return { ranges, text: selection.toString() };
    }
  }

  function formatReference(first, last) {
    return `${path}:${first}${last === first ? "" : `-${last}`}`;
  }

  function selectionReference({ ranges, text }, lineRanges) {
    const matches = lineRanges.filter(({ range: line }) => ranges.some(range =>
      range.compareBoundaryPoints(Range.START_TO_END, line) > 0
      && range.compareBoundaryPoints(Range.END_TO_START, line) < 0));
    if (matches.length) return `${formatReference(matches[0].first, matches.at(-1).last)}\n${text}`;
  }

  function referenceFor({ mediaURL, selectedText, shortcut }) {
    // Math rendering may replace text nodes, so index the current DOM here.
    const { lineRanges, mediaRecords } = indexArticle();
    if (shortcut) {
      const selection = readSelection();
      if (selection?.text.trim()) return selectionReference(selection, lineRanges);

      const media = article.querySelector("img:hover, video:hover, audio:hover")
        ?? document.activeElement?.closest(mediaSelector);
      const record = mediaRecords.get(media);
      if (record) return formatReference(record.first, record.last);
      return;
    }

    if (mediaURL) {
      if (contextMedia && mediaURLs(contextMedia).includes(mediaURL)) {
        const record = mediaRecords.get(contextMedia);
        if (record) return formatReference(record.first, record.last);
      }
      const matches = [...mediaRecords].filter(([media]) => mediaURLs(media).includes(mediaURL));
      if (matches.length === 1) {
        const [, { first, last }] = matches[0];
        return formatReference(first, last);
      }
      return;
    }

    if (!selectedText?.trim()) return;
    const selection = contextSelection ?? readSelection();
    if (selection) return selectionReference(selection, lineRanges);
  }

  const extension = globalThis.browser ?? chrome;
  extension.runtime.onMessage.addListener((message, _sender, respond) => {
    if (message.type !== "copy-markdown-reference") return;
    const reference = referenceFor(message);
    if (!reference) {
      respond(false);
      return;
    }
    navigator.clipboard.writeText(reference).then(() => respond(true), error => {
      console.warn("Couldn’t copy Markdown reference", error);
      respond(false);
    });
    return true;
  });
})();
