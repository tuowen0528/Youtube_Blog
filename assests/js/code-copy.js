document.addEventListener("DOMContentLoaded", () => {
  const codeBlocks = document.querySelectorAll("pre > code");

  codeBlocks.forEach((code) => {
    const container = code.parentElement;

    if (!container) {
      return;
    }

    container.classList.add("code-block");

    if (container.querySelector(".copy-code-button")) {
      return;
    }

    const button = document.createElement("button");
    button.className = "copy-code-button";
    button.type = "button";
    button.textContent = "复制代码";
    button.setAttribute("aria-label", "复制这段代码");

    button.addEventListener("click", async () => {
      const text = code.textContent ?? "";

      try {
        await copyText(text);
        showButtonState(button, "已复制", "copied");
      } catch {
        showButtonState(button, "复制失败", "error");
      }
    });

    container.append(button);
  });
});

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  document.body.append(textarea);
  textarea.select();

  const copied = document.execCommand("copy");
  textarea.remove();

  if (!copied) {
    throw new Error("Copy command failed");
  }
}

function showButtonState(button, label, state) {
  window.clearTimeout(Number(button.dataset.timer));
  button.textContent = label;
  button.dataset.state = state;

  const timer = window.setTimeout(() => {
    button.textContent = "复制代码";
    delete button.dataset.state;
    delete button.dataset.timer;
  }, 1800);

  button.dataset.timer = String(timer);
}
