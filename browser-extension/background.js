const copyCommand = "copy-markdown-reference";
const extension = globalThis.browser ?? chrome;

extension.runtime.onInstalled.addListener(() => {
  extension.contextMenus.create({
    id: copyCommand,
    title: "Copy Markdown reference",
    contexts: ["selection", "image", "video", "audio"],
    documentUrlPatterns: extension.runtime.getManifest().content_scripts[0].matches,
  });
});

async function sendCopyRequest(tab, request) {
  if (tab?.id == null) return;
  try {
    return await extension.tabs.sendMessage(tab.id, {
      type: copyCommand,
      ...request,
    }, { frameId: 0 });
  } catch (error) {
    if (!request.shortcut) console.warn("Couldn’t reach the blog’s Markdown reference data", error);
  }
}

function copyMarkdownReference(info, tab) {
  if (info.menuItemId !== copyCommand) return;
  return sendCopyRequest(tab, {
    mediaURL: info.mediaType ? info.srcUrl : undefined,
    selectedText: info.selectionText,
  });
}

extension.contextMenus.onClicked.addListener(copyMarkdownReference);

async function copyWithShortcut(command, tab) {
  if (command !== copyCommand) return;
  if (tab?.id == null) {
    [tab] = await extension.tabs.query({ active: true, lastFocusedWindow: true });
  }
  return sendCopyRequest(tab, { shortcut: true });
}

extension.commands.onCommand.addListener(copyWithShortcut);
