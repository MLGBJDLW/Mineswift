// Read only the rendered cell states. Never consult the game's board object.
(root, options) => {
    const classic = root.id === "game" && root.querySelector(".square");
    const online = root.id === "game" && root.querySelector("#CellsBlock .cell");
    const rect = e => {
        const b = e.getBoundingClientRect();
        return [b.left, b.top, b.right, b.bottom];
    };
    const rendered = e => e.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})
        && e.getBoundingClientRect().width > 0 && e.getBoundingClientRect().height > 0;
    const text = e => (e?.innerText || e?.getAttribute("aria-label") || "").trim().toLowerCase();
    if (!root.isConnected || !rendered(root)) throw Error("棋盘已隐藏或替换");
    const tiles = [...root.querySelectorAll(classic ? ".square" : online ? "#CellsBlock .cell" : '[role="gridcell"]')].filter(rendered);
    if (!tiles.length || tiles.length > 10000) throw Error("未找到完整棋盘");
    const ordered = tiles.map(e => ({e, b: rect(e)})).sort((a,b) => a.b[1]-b.b[1] || a.b[0]-b.b[0]);
    const [x,y,r,b] = ordered[0].b;
    const cols = ordered.filter(t => Math.abs(t.b[1]-y) < 1).length;
    const rows = tiles.length / cols;
    if (cols < 2 || rows < 2 || !Number.isInteger(rows)) throw Error("棋盘行列不齐");
    const dx = ordered[1].b[0]-x, dy = ordered[cols].b[1]-y;
    if (dx < 5 || dy < 5 || dx > (r-x)*1.35 || dy > (b-y)*1.35) throw Error("格子间距不一致");
    for (const [name, actual] of [["aria-rowcount", rows], ["aria-colcount", cols]]) {
        if (root.hasAttribute(name) && Number(root.getAttribute(name)) !== actual)
            throw Error("声明的行列数与可见格数不符");
    }
    const values = [];
    let exploded = false;
    for (const [i, {e, b: box}] of ordered.entries()) {
        const row = Math.floor(i/cols), col = i%cols;
        if (Math.abs(box[0]-x-col*dx)>1 || Math.abs(box[1]-y-row*dy)>1 ||
            Math.abs(box[2]-box[0]-(r-x))>1 || Math.abs(box[3]-box[1]-(b-y))>1)
            throw Error("棋盘有缺格、错位或格子尺寸不同");
        if (classic && e.id !== (row+1)+"_"+(col+1)) throw Error("格子坐标不连续");
        if (online && e.id !== "cell_"+col+"_"+row) throw Error("格子坐标不连续");
        // Hit-test every centre, including cells clipped by an iframe or a pop-up.
        const hit = document.elementFromPoint((box[0]+box[2])/2, (box[1]+box[3])/2);
        if (!hit || !e.contains(hit)) throw Error("请让整盘可见，并移开遮挡棋盘的弹窗");
        let value;
        if (classic) {
            const states = [...e.classList].filter(s => /^(blank|bombflagged|open[0-8]|bombdeath|bombrevealed|bombmisflagged)$/.test(s));
            if (states.length === 1) {
                const s = states[0];
                if (s === "blank") value = -1;
                else if (s === "bombflagged") value = -2;
                else if (s.startsWith("open")) value = Number(s.slice(4));
                else exploded = true;
            }
        } else if (online) {
            const suffixes = [...e.classList].map(s => s.replace(/^[a-z0-9]+_/, ""));
            if (suffixes.includes("closed") && !suffixes.includes("pressed")) value = suffixes.includes("flag") ? -2 : -1;
            else if (suffixes.includes("opened")) {
                const digits = suffixes.filter(s => /^type[0-8]$/.test(s));
                if (digits.length === 1) value = Number(digits[0].slice(4));
            }
        } else {
            const s = text(e);
            if (/^(covered|closed|hidden|unopened|未开|未翻开)$/.test(s)) value = -1;
            else if (/^(flagged|flag|🚩|旗|已标记)$/.test(s)) value = -2;
            else if (/^(empty|blank|空白)$/.test(s)) value = 0;
            else if (/^[0-8]$/.test(s)) value = Number(s);
            else if (/^(exploded|exploded mine|💥|踩雷)$/.test(s)) exploded = true;
        }
        values.push(value ?? null);
    }
    const face = classic ? root.querySelector("#face") : online ? root.querySelector("#top_area_face") : null;
    let status = "playing";
    if (classic) {
        const states = {facesmile:"playing", facewin:"won", facedead:"lost"};
        status = states[face?.className];
        if (!status) throw Error("表情状态不明确");
    } else if (online) {
        const token = [...(face?.classList ?? [])].find(s => /_top-area-face-(unpressed|win|lose)$/.test(s));
        if (!token || !rendered(face)) throw Error("表情状态不明确");
        status = token.endsWith("-win") ? "won" : token.endsWith("-lose") ? "lost" : "playing";
    } else if (exploded) status = "lost";
    let remaining = null;
    if (status === "playing") {
        if (values.includes(null)) throw Error("棋盘包含无法辨认的格子");
        if (classic || online) {
            const selectors = classic ? ["#mines_hundreds","#mines_tens","#mines_ones"]
                                      : ["#top_area_mines_100","#top_area_mines_10","#top_area_mines_1"];
            const digits = selectors.map(id => {
                const d = root.querySelector(id);
                if (!d || !rendered(d)) throw Error("剩余雷数不可见");
                const m = classic ? /^time([0-9])$/.exec(d.className) : /\b\w+_top-area-num([0-9])\b/.exec(d.className);
                if (!m) throw Error("剩余雷数无法辨认");
                return m[1];
            });
            remaining = Number(digits.join(""));
        } else {
            if (options.mines == null) throw Error("此棋盘需要用 --mines 指定总雷数");
            remaining = options.mines-values.filter(v=>v===-2).length;
            if (values.filter(v=>v<0).length === options.mines) status = "won";
        }
        if (remaining > values.filter(v=>v===-1).length) throw Error("雷数超过未开格数");
    }
    const grid = [x,y,ordered.at(-1).b[2],ordered.at(-1).b[3]];
    const snapshot = {rows,cols,grid,status,remaining,values:status==="playing"?values:[]};
    if (options.target != null) {
        if (JSON.stringify(snapshot)!==JSON.stringify(options.expected)) throw Error("决定落子后棋盘发生变化");
        return options.target === "restart" ? face : ordered[options.target].e;
    }
    return snapshot;
}
