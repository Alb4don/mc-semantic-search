/* Copy-to-clipboard for the agent box phrase. Progressive enhancement:
   the button ships [hidden] and only appears when this script runs, so
   the page stays fully usable without JS. data-copy names the element
   whose text goes to the clipboard; data-label/data-copied-label carry
   the button wording. No dependencies: htmx/Alpine stay untouched. */
(function () {
  "use strict";

  var RESET_MS = 2000;

  function flash(btn, label) {
    btn.textContent = label;
    setTimeout(function () {
      btn.textContent = btn.getAttribute("data-label") || "copiar";
      btn.classList.remove("copied", "failed");
    }, RESET_MS);
  }

  function legacyCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (err) {
      ok = false;
    }
    document.body.removeChild(ta);
    return ok;
  }

  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.hidden = false;
    btn.addEventListener("click", function () {
      var source = document.querySelector(btn.getAttribute("data-copy"));
      var text = source ? source.innerText.trim() : "";
      if (!text) return;
      var done = function (ok) {
        btn.classList.add(ok ? "copied" : "failed");
        flash(btn, ok ? btn.getAttribute("data-copied-label") || "copiado ✓" : "não deu");
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(
          function () {
            done(true);
          },
          function () {
            done(legacyCopy(text));
          }
        );
      } else {
        done(legacyCopy(text));
      }
    });
  });
})();
