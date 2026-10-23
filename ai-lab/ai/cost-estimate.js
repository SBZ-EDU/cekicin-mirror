// Çekicin repair-cost estimate (browser / app / Worker, no dependencies). Data: cost-model.json (every number has a source there).
// items: [{cls: 'dent'|'scratch'|'crack'|'broken_part'|'glass_shatter'|'lamp_broken'|'tire_flat', area: box area as fraction of the photo, part: parts-model class name or part key}]
export function estimateCost(M, items, country = 'TR') {
  const C = M.countries[country] || M.countries.TR, lines = new Map();
  const rank = a => ({ polish: 1, pdr: 2, repaint: 3, plastic_repair: 4, repair_paint: 5, tyre: 6, replace: 9 })[a] || 0;
  for (const it of items) {
    const rules = M.actions[it.cls]; if (!rules) continue;
    const rule = rules.find(r => (it.area || 0) <= r.max_area) || rules[rules.length - 1];
    let p = it.part && (M.part_of[it.part] || (M.parts[it.part] ? it.part : null)), assumed = false;
    if (!p || (it.cls === 'glass_shatter' && !M.parts[p].glass) || (it.cls === 'lamp_broken' && !M.parts[p].lamp)) {
      // 1.1: a close-up usually shows one area -> reuse the body part already chosen for another damage in this photo, else the class default
      const body = c => !['glass_shatter', 'lamp_broken', 'tire_flat'].includes(c);
      const known = body(it.cls) && items.map(o => o.part && (M.part_of[o.part] || (M.parts[o.part] ? o.part : null))).find(q => q && !M.parts[q].glass && !M.parts[q].lamp && q !== 'tyre');
      const first = body(it.cls) && items.find(o => body(o.cls) && o.cls !== 'broken_part');
      p = known || (first ? M.default_part[first.cls] : M.default_part[it.cls]); assumed = true;
    }
    // broken_part has ~50 % precision on real-world photos (2.3.1 audit): never a sure line. If its part is glass / a lamp that
    // another detected type already prices, drop it; otherwise show it as a conditional extra, outside the total.
    const conditional = it.cls === 'broken_part';
    if (conditional && (assumed || M.parts[p].glass || M.parts[p].lamp) && items.some(o => o.cls === 'glass_shatter' || o.cls === 'lamp_broken')) continue;
    const P = M.parts[p], key = (conditional ? 'cond:' : '') + p, old = lines.get(key);
    if (old && rank(old.action) >= rank(rule.action)) continue;
    let lo = 0, hi = 0;
    const hb = rule.part ? P.replace_h : rule.body * P.body_h;                       // body / fitting hours
    lo += hb * C.body_rate[0]; hi += hb * C.body_rate[1];
    if (rule.paint_frac > 0) {
      if (C.ato_paint && C.ato_paint[p]) { const v = C.ato_paint[p][rule.part ? 0 : 1] * rule.paint_frac; lo += v * C.market_floor; hi += v; }
      else if (C.paint_rate) { const h = P.paint_h * rule.paint_frac; lo += h * (C.paint_rate[0] + C.paint_material[0]); hi += h * (C.paint_rate[1] + C.paint_material[1]); }
    }
    // independent-shop floor (TR): the ATO tariff is a legal maximum, so a repair line's low end is the market quote floor
    const ml = C.market_low && C.market_low[rule.action] && C.market_low[rule.action][P.plastic ? 'plastic' : 'metal'];
    if (ml && !rule.part) lo = Math.min(lo, ml);
    let partSrc = null;
    if (rule.part && C.parts[p]) { lo += C.parts[p].min; hi += C.parts[p].max; partSrc = C.parts[p]; }
    lines.set(key, { part: p, part_tr: P.tr, part_en: P.en, cls: it.cls, action: rule.action, action_tr: rule.tr, action_en: rule.en,
                     min: Math.round(lo), max: Math.round(hi), part_price: partSrc, hours: Math.round(hb * 10) / 10,
                     assumed, conditional });
  }
  const all = [...lines.values()].sort((a, b) => b.max - a.max);
  const L = all.filter(l => !l.conditional), X = all.filter(l => l.conditional);   // shown as "if replacement is needed", never added to the total
  const total = { min: L.reduce((s, l) => s + l.min, 0), max: L.reduce((s, l) => s + l.max, 0) };
  return { country, currency: C.currency, symbol: C.symbol, vat: C.vat, vehicle: C.vehicle, lines: L, conditional: X, total,
           assumed_parts: L.some(l => l.assumed),
           note_tr: C.note_tr, note_en: C.note_en, sources: C.sources.map(k => M.sources[k]) };
}
export function fmtMoney(v, cur, lang = 'tr') {
  const loc = { TRY: 'tr-TR', EUR: 'de-DE', GBP: 'en-GB', USD: 'en-US' }[cur] || (lang === 'tr' ? 'tr-TR' : 'en-GB');
  const r = v >= 10000 ? Math.round(v / 500) * 500 : v >= 1000 ? Math.round(v / 50) * 50 : Math.round(v / 5) * 5;
  return new Intl.NumberFormat(loc, { style: 'currency', currency: cur, maximumFractionDigits: 0 }).format(r);
}
