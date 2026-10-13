const state = {
    dataset: "",
    curve: "",
    recover: "",
    source: "",
    q: "",
    offset: 0,
    total: 0,
    loading: false,
};

const $ = (id) => document.getElementById(id);
const els = {
    dataset: $("dataset"),
    curve: $("filter-curve"),
    recover: $("filter-recover"),
    source: $("filter-source"),
    q: $("filter-q"),
    grid: $("grid"),
    count: $("result-count"),
    more: $("btn-more"),
    modal: $("modal"),
    modalImg: $("modal-img"),
    modalMeta: $("modal-meta"),
    toast: $("toast"),
};

function imageUrl(item) {
    return `/api/image?source=${encodeURIComponent(item.source)}&name=${encodeURIComponent(item.image)}`;
}

function toast(msg) {
    els.toast.textContent = msg;
    els.toast.classList.remove("hidden");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => els.toast.classList.add("hidden"), 2200);
}

async function getJson(url) {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `请求失败 (${res.status})`);
    return data;
}

function fillSelect(select, values, keepFirst = true) {
    const first = keepFirst ? select.querySelector("option").outerHTML : "";
    select.innerHTML = first + values.map((v) => `<option value="${v}">${v}</option>`).join("");
}

async function init() {
    try {
        const datasets = await getJson("/api/datasets");
        if (!datasets.length) {
            toast("未发现任何数据集");
            return;
        }
        fillSelect(els.dataset, datasets.map((d) => `${d.name}（${d.count}）`), false);
        els.dataset.value = els.dataset.querySelector("option").value;
        els.dataset.addEventListener("change", () => {
            state.dataset = els.dataset.value.split("（")[0];
            applyDataset(datasets.find((d) => d.name === state.dataset));
        });

        state.dataset = els.dataset.value.split("（")[0];
        applyDataset(datasets.find((d) => d.name === state.dataset));
    } catch (e) {
        toast(e.message);
    }
}

function applyDataset(dataset) {
    fillSelect(els.curve, dataset.curves);
    fillSelect(els.recover, dataset.recovers);
    fillSelect(els.source, dataset.sources);
    resetFilters(false);
}

function resetFilters(reload = true) {
    els.curve.value = "";
    els.recover.value = "";
    els.source.value = "";
    els.q.value = "";
    Object.assign(state, { curve: "", recover: "", source: "", q: "" });
    if (reload) reload();
}

function buildUrl(appendOffset) {
    const p = new URLSearchParams({ dataset: state.dataset });
    if (state.curve) p.set("curve", state.curve);
    if (state.recover) p.set("recover", state.recover);
    if (state.source) p.set("source", state.source);
    if (state.q) p.set("q", state.q);
    if (appendOffset) {
        p.set("offset", state.offset);
        p.set("limit", 60);
    }
    return `/api/items?${p.toString()}`;
}

async function reload() {
    state.offset = 0;
    els.grid.innerHTML = "";
    await loadPage();
}

async function loadPage() {
    if (state.loading) return;
    state.loading = true;
    els.more.disabled = true;
    try {
        const data = await getJson(buildUrl(true));
        state.total = data.total;
        state.offset += data.items.length;
        renderCount(data);
        renderCards(data.items);
        els.more.style.display = state.offset < state.total ? "" : "none";
        if (!state.total) {
            els.grid.innerHTML = `<div class="empty">没有符合筛选条件的数据</div>`;
        }
    } catch (e) {
        toast(e.message);
    } finally {
        state.loading = false;
        els.more.disabled = false;
    }
}

function renderCount(data) {
    const shown = Math.min(state.offset, state.total);
    els.count.innerHTML = `共 <b>${state.total}</b> 条 · 已显示 <b>${shown}</b> 条`;
}

function renderCards(items) {
    const frag = document.createDocumentFragment();
    for (const item of items) {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
            <img class="card-img" loading="lazy" src="${imageUrl(item)}" alt="${item.image}">
            <div class="card-body">
                <div class="badges">
                    <span class="badge curve-${item.curve}">curve: ${item.curve}</span>
                    <span class="badge recover-${item.recover}">${item.recover}</span>
                    <span class="badge source">${item.source}</span>
                </div>
                <div class="card-name" title="${item.image}">#${item.index} · ${item.image}</div>
            </div>`;
        card.addEventListener("click", () => openModal(item));
        frag.appendChild(card);
    }
    els.grid.appendChild(frag);
}

function openModal(item) {
    els.modalImg.src = imageUrl(item);
    els.modalMeta.innerHTML = `
        <span class="badge curve-${item.curve}">curve: ${item.curve}</span>
        <span class="badge recover-${item.recover}">${item.recover}</span>
        <span class="badge source">${item.source}</span>
        <span class="file">#${item.index} · ${item.image}</span>`;
    els.modal.classList.remove("hidden");
}

function closeModal() {
    els.modal.classList.add("hidden");
    els.modalImg.src = "";
}

/* ---------- 事件绑定 ---------- */
let searchTimer;
els.q.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.q = els.q.value.trim(); reload(); }, 300);
});
els.curve.addEventListener("change", () => { state.curve = els.curve.value; reload(); });
els.recover.addEventListener("change", () => { state.recover = els.recover.value; reload(); });
els.source.addEventListener("change", () => { state.source = els.source.value; reload(); });
$("btn-reset").addEventListener("click", () => resetFilters(true));
els.more.addEventListener("click", loadPage);
els.modal.querySelector(".modal-backdrop").addEventListener("click", closeModal);
els.modal.querySelector(".modal-close").addEventListener("click", closeModal);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

init();
