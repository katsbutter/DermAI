/**
 * DermAI — JavaScript principal
 * Cours : Introduction à l'IA — ENSTAB | Dr. Amira Echtioui
 */

document.addEventListener("DOMContentLoaded", () => {

    // ── Horloge en temps réel ──────────────────────────────────────────
    const clockEl = document.getElementById("current-time");
    if (clockEl) {
        const updateClock = () => {
            const now = new Date();
            clockEl.textContent = now.toLocaleTimeString("fr-FR", {
                hour:   "2-digit",
                minute: "2-digit",
                second: "2-digit",
            });
        };
        updateClock();
        setInterval(updateClock, 1000);
    }

    // ── Toggle sidebar (mobile) ────────────────────────────────────────
    const toggleBtn = document.getElementById("sidebarToggle");
    const sidebar   = document.getElementById("sidebar");

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener("click", () => {
            sidebar.classList.toggle("open");
        });

        // Fermer en cliquant hors de la sidebar
        document.addEventListener("click", (e) => {
            if (
                sidebar.classList.contains("open") &&
                !sidebar.contains(e.target) &&
                !toggleBtn.contains(e.target)
            ) {
                sidebar.classList.remove("open");
            }
        });
    }

    // ── Auto-dismiss des alertes flash après 5 secondes ──────────────
    const alerts = document.querySelectorAll(".alert.alert-dismissible");
    alerts.forEach((alert) => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 5000);
    });

    // ── Activation des tooltips Bootstrap ─────────────────────────────
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => {
        new bootstrap.Tooltip(el);
    });

});
