/* Copy-to-clipboard for the agent box phrase. Progressive enhancement:
   the corner icon chip ships [hidden] and only appears when this script
   runs, so the page stays fully usable without JS. data-copy names the
   element whose text goes to the clipboard. Feedback is class-driven —
   copy icon swaps to a check, and the sr-only role=status span announces
   the result. No dependencies: htmx/Alpine stay untouched. */
(function () {
  "use strict";

  var RESET_MS = 2000;

  function flash(btn, ok) {
    btn.classList.add(ok ? "copied" : "failed");
    var state = btn.parentNode.querySelector(".copy-state");
    if (state) state.textContent = ok ? "copiado" : "não deu para copiar";
    setTimeout(function () {
      btn.classList.remove("copied", "failed");
      if (state) state.textContent = "";
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
        flash(btn, ok);
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
