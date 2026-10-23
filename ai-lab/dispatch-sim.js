/* Çekicin tow-dispatch simulator (Monte Carlo, discrete events). Same incidents, same trucks, two policies:
   base    = call centre: manual dispatch delay; only a share of trucks/workshops is known; nearest of those.
   cekicin = app: short automatic dispatch; nearest available truck (by arrival time) and nearest workshop.
   All inputs are assumptions until replaced by pilot data. */
(function (root) {
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function simulate(p, seed) {
    const R = rng(seed || 7), W = p.W, H = p.H, minutes = p.days * 1440;
    const pt = () => ({ x: R() * W, y: R() * H });
    const trucks0 = Array.from({ length: p.trucks }, pt), services = Array.from({ length: p.services }, pt);
    const inc = []; for (let t = 0; ; ) { t += -Math.log(1 - R()) * 60 / p.perHour; if (t > minutes) break; inc.push({ t, ...pt(), block: R() < p.block, u: [] }); }
    // per-incident random draws for the call-centre subsets, shared by nothing else (keeps runs reproducible)
    inc.forEach(e => { e.tk = trucks0.map(() => R()); e.sv = services.map(() => R()); });
    const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y) * p.circuity;
    function run(policy) {
      const tr = trucks0.map(q => ({ x: q.x, y: q.y, free: 0 }));
      const out = { wait: [], total: [], empty: 0, loaded: 0, hazard: 0, n: 0 };
      for (const e of inc) {
        const base = policy === 'base', ready = e.t + (base ? p.ccMin : p.appMin);
        let best = null, bestArr = Infinity;
        tr.forEach((q, i) => {
          if (base && e.tk[i] > p.share && i !== 0) return;          // call centre only reaches some firms (truck 0 = its default contractor)
          const arr = Math.max(q.free, ready) + d(q, e) / p.speed * 60;
          if (arr < bestArr) { bestArr = arr; best = q; }
        });
        let svc = null, sd = Infinity;
        // workshop: often chosen by the insurer/customer, so the app may only pick the nearest one when allowed (svcNearest)
        services.forEach((s, i) => { if ((base || !p.svcNearest) && e.sv[i] > p.share && i !== 0) return; const k = d(e, s); if (k < sd) { sd = k; svc = s; } });
        const empty = d(best, e), wait = bestArr - e.t, done = bestArr + p.loadMin + sd / p.speed * 60;
        best.free = done; best.x = svc.x; best.y = svc.y;
        out.wait.push(wait); out.total.push(done - e.t); out.empty += empty; out.loaded += sd; out.n++;
        if (e.block) out.hazard += wait + p.loadMin;
      }
      const mean = a => a.reduce((s, v) => s + v, 0) / a.length, q = (a, f) => a.slice().sort((x, y) => x - y)[Math.floor(f * (a.length - 1))];
      return { n: out.n, wait: mean(out.wait), wait90: q(out.wait, 0.9), total: mean(out.total), emptyKm: out.empty / out.n, loadedKm: out.loaded / out.n,
               hazardPerBlock: out.hazard / Math.max(1, inc.filter(e => e.block).length) };
    }
    return { base: run('base'), cekicin: run('cekicin'), incidents: inc.length };
  }
  function impact(r, p, ef) {      // per month, from the simulated per-job averages
    const jobs = p.perHour * 24 * 30, f = s => {
      const towKg = (s.emptyKm + s.loadedKm) * ef.tow_truck_km, idleKg = p.idleShare * (s.wait / 60) * p.idleLph * ef.diesel_litre;
      return { towT: jobs * towKg / 1000, idleT: jobs * idleKg / 1000, totT: jobs * (towKg + idleKg) / 1000, emptyKmMonth: jobs * s.emptyKm,
               hazardMinMonth: jobs * p.block * s.hazardPerBlock };
    };
    const b = f(r.base), c = f(r.cekicin), red = (x, y) => (x - y) / x * 100;
    const savedHazard = r.base.hazardPerBlock - r.cekicin.hazardPerBlock;
    return { jobs, b, c, waitRed: red(r.base.wait, r.cekicin.wait), totalRed: red(r.base.total, r.cekicin.total), emptyRed: red(r.base.emptyKm, r.cekicin.emptyKm),
             co2Red: red(b.totT, c.totT), co2SavedT: b.totT - c.totT, hazardMinSaved: savedHazard,
             secondaryRel: 100 * 0.028 * savedHazard / (1 + 0.028 * r.base.hazardPerBlock) };
    // FHWA rule of thumb: secondary-crash likelihood +2.8 % per minute the incident remains a hazard -> P(T) ~ 1 + 0.028 T; relative cut vs baseline exposure
  }
  const DEFAULTS = { W: 45, H: 25, trucks: 60, services: 80, perHour: 10, speed: 22, circuity: 1.35, ccMin: 20, appMin: 2, share: 0.15,
                     loadMin: 15, block: 0.35, idleShare: 0.5, idleLph: 0.8, days: 7, svcNearest: false };
  const EF = { tow_truck_km: 0.70603, diesel_car_km: 0.17336, diesel_litre: 2.51 };
  root.CekicinSim = { simulate, impact, DEFAULTS, EF };
  if (typeof module !== 'undefined') module.exports = root.CekicinSim;
})(typeof window !== 'undefined' ? window : globalThis);
