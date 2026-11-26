// ============================================================================
//  Giant-impact Moon formation — 2D toy simulation core
//  Units: G = 1, M_earth = 1, R_earth = 1  ->  1 time unit ≈ 805 s (13.4 min)
//                                            1 velocity unit ≈ 7.91 km/s
//
//  Phase 1 (impact, ~1 day): every particle is a parcel of rock or iron.
//     - self-gravity via a Barnes–Hut quadtree (O(N log N))
//     - smoothed particle hydrodynamics (SPH), the method Canup (2012) used,
//       with a simple compressible EOS, shock heating and slow cooling
//  Phase 2 (accretion, weeks–months): the merged planet is replaced by one
//     body; the remaining debris disk orbits it. Debris outside the Roche
//     limit sticks together when gravitationally bound -> moonlets -> Moon.
// ============================================================================
(function (root) {
  'use strict';
  const T_UNIT_S = 805.5;          // seconds per time unit
  const V_UNIT_KMS = 7.91;         // km/s per velocity unit
  const M_LUNAR = 0.0123;          // lunar mass in Earth masses
  // Temperature = T_BASE + u * T_PER_U, with u the specific internal energy that the SPH
  // energy equation tracks (shock heating, compression work, cooling). T_BASE is the cold
  // starting temperature (bottom of the paper's Fig. 1 colour bar). T_PER_U converts
  // energy to temperature with an effective heat capacity of ~8 kJ/kg/K, much larger
  // than a bare c_v because real shock energy also goes into melting and vaporising rock
  // (the paper's tabulated equations of state include this). It is a calibration, so the
  // absolute temperatures are indicative; the pattern of hot versus cold is what is real.
  const T_BASE = 2500, T_PER_U = 7800;

  // particle kinds
  const TGT_MANTLE = 0, TGT_CORE = 1, IMP_MANTLE = 2, IMP_CORE = 3, MOONLET = 4;

  function Sim(opts) {
    this.o = Object.assign({
      N: 2000,          // particle count for 1 Earth mass of material
      gamma: 0.45,      // impactor mass / total mass   (Canup 2012: 0.40–0.45)
      b: 0.55,          // impact parameter sin(angle)   (Canup 2012: 0.35–0.7)
      vratio: 1.1,      // v_imp / v_esc                 (Canup 2012: 1.0–1.6)
      dt: 0.005,        // max timestep, impact phase (adaptive CFL below this)
      dt2: 0.01,        // timestep, accretion phase
      tSwitch: 110,     // switch to accretion phase (~1 day)
      gravDt: 0.015,    // how often the gravity tree is rebuilt in the impact phase (time units)
      gravExtrap: true, // extrapolate gravity between rebuilds (false = hold it constant)
      gth: 0.3,         // thermal pressure coefficient (Grüneisen-like)
      tcool: 15,        // radiative cooling time of hot material (0 = off)
      tCirc: 20,        // disk orbit circularisation time (phase 2)
      tVisc: 400,       // inner-disk inflow time inside the Roche limit (phase 2)
      hill: 3.5,        // clumping range in Hill radii (phase 2)
      seed: 1,
    }, opts || {});
    this.t = 0;
    this.phase = 0;     // 0 = setup, 1 = impact, 2 = accretion
    this.log = [];
    this._rng = mulberry32(this.o.seed);
    this.init();
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------------------- storage
  Sim.prototype.alloc = function (cap) {
    this.cap = cap;
    const F = Float64Array;
    this.x = new F(cap); this.y = new F(cap);
    this.vx = new F(cap); this.vy = new F(cap);
    this.ax = new F(cap); this.ay = new F(cap);
    this.m = new F(cap); this.r = new F(cap);
    this.u = new F(cap);            // specific internal energy (heat)
    this.ftar = new F(cap);         // mass fraction derived from the target
    this.kind = new Uint8Array(cap);
    this.n = 0;
  };

  Sim.prototype.add = function (x, y, vx, vy, m, r, kind, ftar) {
    const i = this.n++;
    this.x[i] = x; this.y[i] = y; this.vx[i] = vx; this.vy[i] = vy;
    this.m[i] = m; this.r[i] = r; this.kind[i] = kind; this.u[i] = 0;
    this.ftar[i] = ftar;
    return i;
  };

  Sim.prototype.remove = function (i) {   // swap-with-last
    const j = --this.n;
    if (i === j) return;
    for (const a of [this.x, this.y, this.vx, this.vy, this.ax, this.ay, this.m, this.r, this.u, this.ftar, this.kind, this.rho0, this.gax, this.gay, this.du, this.rho]) if (a) a[i] = a[j];
  };

  // ---------------------------------------------------------------- set-up
  Sim.prototype.init = function () {
    const o = this.o;
    this.alloc(Math.ceil(o.N * 1.25) + 50);
    // hex lattice spacing chosen so 1 Earth mass of particles fills radius 1
    const s = Math.sqrt(Math.PI / (0.866 * o.N));
    this.s = s; this.r0 = s / 2;
    this.m0 = 1 / (o.N * 1.15);            // mantle particle mass (core = 1.6x)
    this.eps = 0.6 * s;                    // gravitational softening
    const Mt = 1 - o.gamma, Mi = o.gamma;
    // build both bodies far apart and let them settle under their own gravity
    this.buildBody(-15, 0, Mt, TGT_MANTLE, TGT_CORE, 1);
    this.nT = this.n;
    this.buildBody(15, 0, Mi, IMP_MANTLE, IMP_CORE, 0);
    this.relax(500);
    this.place();
    this.phase = 1;
  };

  Sim.prototype.buildBody = function (cx, cy, M, km, kc, ftar) {
    const R = Math.sqrt(M), s = this.s, rng = this._rng;
    for (let row = -Math.ceil(R / (s * 0.866)) - 1; row <= Math.ceil(R / (s * 0.866)) + 1; row++) {
      const yy = row * s * 0.866, off = (row & 1) ? s / 2 : 0;
      for (let col = -Math.ceil(R / s) - 1; col <= Math.ceil(R / s) + 1; col++) {
        const xx = col * s + off;
        const rr = Math.hypot(xx, yy);
        if (rr > R) continue;
        const core = rr < 0.5 * R;
        const jx = (rng() - 0.5) * 0.15 * s, jy = (rng() - 0.5) * 0.15 * s;
        this.add(cx + xx + jx, cy + yy + jy, 0, 0, this.m0 * (core ? 1.6 : 1), this.r0, core ? kc : km, ftar);
      }
    }
  };

  Sim.prototype.relax = function (steps) {
    this.sphSetup();
    // reference density = lattice density (core particles are 1.6x heavier)
    const lat = this.m0 / (0.866 * this.s * this.s);
    for (let i = 0; i < this.n; i++) this.rho0[i] = lat * (this.kind[i] === TGT_CORE || this.kind[i] === IMP_CORE ? 1.6 : 1);
    for (let k = 0; k < steps; k++) {
      this.stepImpact(this.o.dt);
      const damp = k < steps * 0.8 ? 0.97 : 0.995;
      for (let i = 0; i < this.n; i++) { this.vx[i] *= damp; this.vy[i] *= damp; this.u[i] = 0; }
    }
  };

  Sim.prototype.bodyStats = function (from, to) {
    let M = 0, X = 0, Y = 0, VX = 0, VY = 0;
    for (let i = from; i < to; i++) { const m = this.m[i]; M += m; X += m * this.x[i]; Y += m * this.y[i]; VX += m * this.vx[i]; VY += m * this.vy[i]; }
    X /= M; Y /= M; VX /= M; VY /= M;
    let R = 0;
    for (let i = from; i < to; i++) R = Math.max(R, Math.hypot(this.x[i] - X, this.y[i] - Y));
    return { M, X, Y, VX, VY, R: R + this.r0 };
  };

  // put the two bodies on a hyperbolic/parabolic approach that hits with the
  // requested impact parameter b and speed v_imp = vratio * v_esc
  Sim.prototype.place = function () {
    const o = this.o;
    const T = this.bodyStats(0, this.nT), I = this.bodyStats(this.nT, this.n);
    const M = T.M + I.M, dc = T.R + I.R;
    const vesc = Math.sqrt(2 * M / dc), vimp = o.vratio * vesc;
    this.vesc = vesc; this.vimp = vimp;
    // relative state at contact (impactor relative to target)
    let rx = dc * Math.sqrt(1 - o.b * o.b), ry = dc * o.b, vx = -vimp, vy = 0;
    // integrate the two-body problem backwards until well separated
    const h = -0.002;
    let tb = 0;
    while (Math.hypot(rx, ry) < 1.6 * dc && tb > -20) {
      // RK2 midpoint
      const a1 = acc(rx, ry);
      const mx = rx + 0.5 * h * vx, my = ry + 0.5 * h * vy;
      const mvx = vx + 0.5 * h * a1[0], mvy = vy + 0.5 * h * a1[1];
      const a2 = acc(mx, my);
      rx += h * mvx; ry += h * mvy; vx += h * a2[0]; vy += h * a2[1];
      tb += h;
    }
    function acc(x, y) { const d = Math.hypot(x, y); const f = -M / (d * d * d); return [f * x, f * y]; }
    const ft = -I.M / M, fi = T.M / M;
    for (let i = 0; i < this.n; i++) {
      const isT = i < this.nT, S = isT ? T : I, f = isT ? ft : fi;
      this.x[i] += -S.X + f * rx; this.y[i] += -S.Y + f * ry;
      this.vx[i] = f * vx; this.vy[i] = f * vy;
    }
    this.t = tb; // negative: time until contact
    this._gravT = NaN; this._haveAcc = false;   // forces from the relaxed, separated bodies are stale now
    this.log.push(`contact in ${(-tb * T_UNIT_S / 60).toFixed(0)} min, v_imp = ${(vimp * V_UNIT_KMS).toFixed(1)} km/s (toy units)`);
  };

  // ---------------------------------------------------------------- gravity
  Sim.prototype.buildTree = function () {
    const n = this.n, maxNodes = 16 * n + 64;
    if (!this.tMass || this.tMass.length < maxNodes) {
      this.tMass = new Float64Array(maxNodes); this.tMx = new Float64Array(maxNodes); this.tMy = new Float64Array(maxNodes);
      this.tCx = new Float64Array(maxNodes); this.tCy = new Float64Array(maxNodes); this.tH = new Float64Array(maxNodes);
      this.tBody = new Int32Array(maxNodes); this.tChild = new Int32Array(maxNodes);
      this.stack = new Int32Array(maxNodes);
    }
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = this.x[i], y = this.y[i];
      if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    const tM = this.tMass, tMx = this.tMx, tMy = this.tMy, tCx = this.tCx, tCy = this.tCy, tH = this.tH, tB = this.tBody, tC = this.tChild;
    let nn = 1;
    const minH = 0.05 * this.s;
    tCx[0] = (minx + maxx) / 2; tCy[0] = (miny + maxy) / 2; tH[0] = Math.max(maxx - minx, maxy - miny) / 2 + 1e-6;
    tM[0] = tMx[0] = tMy[0] = 0; tB[0] = -1; tC[0] = -1;
    const X = this.x, Y = this.y, Ms = this.m;
    for (let b = 0; b < n; b++) {
      const bx = X[b], by = Y[b], bm = Ms[b];
      let node = 0;
      for (;;) {
        tM[node] += bm; tMx[node] += bm * bx; tMy[node] += bm * by;
        if (tC[node] < 0) {                     // leaf
          if (tB[node] === -1) { tB[node] = b; break; }
          if (tH[node] < minH || nn + 4 > maxNodes) break; // near-coincident: bucket
          // split: move existing body into a child
          const c = tB[node]; tB[node] = -2;
          const first = nn; nn += 4; tC[node] = first;
          const h2 = tH[node] / 2;
          for (let q = 0; q < 4; q++) {
            const k = first + q;
            tCx[k] = tCx[node] + ((q & 1) ? h2 : -h2); tCy[k] = tCy[node] + ((q & 2) ? h2 : -h2); tH[k] = h2;
            tM[k] = tMx[k] = tMy[k] = 0; tB[k] = -1; tC[k] = -1;
          }
          const qc = (X[c] > tCx[node] ? 1 : 0) | (Y[c] > tCy[node] ? 2 : 0);
          const kc = first + qc, cm = Ms[c];
          tM[kc] = cm; tMx[kc] = cm * X[c]; tMy[kc] = cm * Y[c]; tB[kc] = c;
        }
        const q = (bx > tCx[node] ? 1 : 0) | (by > tCy[node] ? 2 : 0);
        node = tC[node] + q;
      }
    }
    this.nNodes = nn;
  };

  Sim.prototype.gravityTree = function () {
    const n = this.n, theta2 = 0.7 * 0.7, eps2 = this.eps * this.eps;
    const tM = this.tMass, tMx = this.tMx, tMy = this.tMy, tH = this.tH, tB = this.tBody, tC = this.tChild, st = this.stack;
    for (let i = 0; i < n; i++) {
      const xi = this.x[i], yi = this.y[i];
      let ax = 0, ay = 0, sp = 0;
      st[sp++] = 0;
      while (sp > 0) {
        const nd = st[--sp];
        const m = tM[nd]; if (m === 0) continue;
        if (tB[nd] === i) continue;
        const cx = tMx[nd] / m, cy = tMy[nd] / m;
        const dx = cx - xi, dy = cy - yi, d2 = dx * dx + dy * dy;
        const w = 2 * tH[nd];
        if (tC[nd] < 0 || w * w < theta2 * d2) {
          const r2 = d2 + eps2, inv = m / (r2 * Math.sqrt(r2));
          ax += dx * inv; ay += dy * inv;
        } else {
          const c = tC[nd];
          st[sp++] = c; st[sp++] = c + 1; st[sp++] = c + 2; st[sp++] = c + 3;
        }
      }
      this.ax[i] = ax; this.ay[i] = ay;
    }
  };

  // ------------------------------------------------- phase-2 contact model
  // Debris clumps in the disk collide as soft spheres (spring + dashpot).
  const OMEGA = 60, ZETA = 0.35;
  function contact(S, i, j) {
    const dx = S.x[j] - S.x[i], dy = S.y[j] - S.y[i];
    const rc = S.r[i] + S.r[j];
    const d2 = dx * dx + dy * dy;
    if (d2 >= rc * rc || d2 === 0) return;
    const d = Math.sqrt(d2), nx = dx / d, ny = dy / d;
    const mi = S.m[i], mj = S.m[j], mu = mi * mj / (mi + mj);
    const k = mu * OMEGA * OMEGA, c = 2 * ZETA * mu * OMEGA;
    const vn = (S.vx[j] - S.vx[i]) * nx + (S.vy[j] - S.vy[i]) * ny;
    let f = k * (rc - d) - c * vn;
    if (f < 0) f = 0;
    S.ax[i] -= f * nx / mi; S.ay[i] -= f * ny / mi;
    S.ax[j] += f * nx / mj; S.ay[j] += f * ny / mj;
  }

  // ------------------------------------------------- phase-1 SPH hydrodynamics
  // 2D cubic-spline kernel, summation density, simple "stiffened" EOS:
  //    P = max(0, c0^2 (rho - rho0)) + GAMMA_TH * rho * u
  // (cold compression term + thermal term, so shock-heated rock puffs up like
  //  vapour), Monaghan artificial viscosity, and an internal-energy equation.
  const C0 = 1.2, AV_A = 1.0, AV_B = 2.0, HFAC = 1.2;
  let GAMMA_TH = 0.6;

  Sim.prototype.sphSetup = function () {
    GAMMA_TH = this.o.gth;
    this.h = HFAC * this.s;
    const h = this.h;
    this.kSig = 10 / (7 * Math.PI * h * h);
    if (!this.rho || this.rho.length < this.cap) {
      this.rho = new Float64Array(this.cap); this.P = new Float64Array(this.cap); this.cs = new Float64Array(this.cap);
      this.du = new Float64Array(this.cap); this.gax = new Float64Array(this.cap); this.gay = new Float64Array(this.cap);
      this.pgx = new Float64Array(this.cap); this.pgy = new Float64Array(this.cap);
      this.rho0 = new Float64Array(this.cap);
    }
  };
  Sim.prototype.W = function (r) {
    const q = r / this.h;
    if (q < 1) return this.kSig * (1 - 1.5 * q * q + 0.75 * q * q * q);
    if (q < 2) { const t = 2 - q; return this.kSig * 0.25 * t * t * t; }
    return 0;
  };
  Sim.prototype.dW = function (r) {       // dW/dr
    const q = r / this.h, f = this.kSig / this.h;
    if (q < 2 / 3) return -f;               // Thomas–Couchman: no pairing
    if (q < 1) return f * (-3 * q + 2.25 * q * q);
    if (q < 2) { const t = 2 - q; return -0.75 * f * t * t; }
    return 0;
  };

  // Exact cell list for neighbour search. Cells are 2h wide, so every neighbour of a
  // particle lies in its own cell or one of the 8 around it. Occupied cells live in an
  // open-addressing table keyed on the cell coordinates (no false sharing between
  // cells), particles are counting-sorted by cell, and each cell stores the ids of its
  // 4 "forward" neighbour cells so every pair of cells is visited exactly once.
  const FWD = [[1, 0], [-1, 1], [0, 1], [1, 1]];
  Sim.prototype.buildGrid = function () {
    const n = this.n, inv = 1 / (2 * this.h), cap = this.cap;
    let size = 4, bits = 2; while (size < 4 * n) { size <<= 1; bits++; }
    const mask = size - 1, shift = 32 - bits;
    if (!this.gKey || this.gKey.length < size) { this.gKey = new Int32Array(size); this.gVal = new Int32Array(size); }
    if (!this.cNbr || this.cStart.length < cap + 2) {
      this.gCid = new Int32Array(cap); this.gOrd = new Int32Array(cap);
      this.cStart = new Int32Array(cap + 2); this.cFill = new Int32Array(cap + 1);
      this.cCx = new Int32Array(cap); this.cCy = new Int32Array(cap); this.cNbr = new Int32Array(4 * cap);
    }
    const gKey = this.gKey, gVal = this.gVal, cid = this.gCid, ord = this.gOrd, cStart = this.cStart, cFill = this.cFill;
    const cCx = this.cCx, cCy = this.cCy, cNbr = this.cNbr, X = this.x, Y = this.y;
    gKey.fill(-1, 0, size);
    let nc = 0;
    cStart.fill(0, 0, n + 2);
    for (let i = 0; i < n; i++) {
      const cx = Math.floor(X[i] * inv), cy = Math.floor(Y[i] * inv);
      const key = ((cx + 16384) << 15) | (cy + 16384);
      let p = Math.imul(key, -1640531535) >>> shift;
      while (gKey[p] !== -1 && gKey[p] !== key) p = (p + 1) & mask;
      let c;
      if (gKey[p] === -1) { gKey[p] = key; c = gVal[p] = nc; cCx[nc] = cx; cCy[nc] = cy; nc++; } else c = gVal[p];
      cid[i] = c; cStart[c + 1]++;
    }
    for (let c = 0; c < nc; c++) { cStart[c + 1] += cStart[c]; cFill[c] = cStart[c]; }
    for (let i = 0; i < n; i++) ord[cFill[cid[i]]++] = i;
    for (let c = 0; c < nc; c++) {
      for (let k = 0; k < 4; k++) {
        const key = ((cCx[c] + FWD[k][0] + 16384) << 15) | (cCy[c] + FWD[k][1] + 16384);
        let p = Math.imul(key, -1640531535) >>> shift, found = -1;
        while (gKey[p] !== -1) { if (gKey[p] === key) { found = gVal[p]; break; } p = (p + 1) & mask; }
        cNbr[4 * c + k] = found;
      }
    }
    this.nCells = nc;
  };

  Sim.prototype.sphForces = function () {
    const n = this.n, h = this.h, h2x4 = 4 * h * h, ih = 1 / h, sig = this.kSig, fsig = sig * ih;
    const X = this.x, Y = this.y, VX = this.vx, VY = this.vy, M = this.m, U = this.u;
    const rho = this.rho, P = this.P, cs = this.cs, AX = this.ax, AY = this.ay, DU = this.du, R0 = this.rho0;
    this.buildGrid();
    // 1) neighbour pairs + density (each pair found once, via the cell list)
    if (!this.pI || this.pI.length < n * 40) {
      this.pI = new Int32Array(n * 40); this.pJ = new Int32Array(n * 40);
      this.pDx = new Float64Array(n * 40); this.pDy = new Float64Array(n * 40); this.pR = new Float64Array(n * 40);
    }
    let np = 0;
    let pI = this.pI, pJ = this.pJ, pDx = this.pDx, pDy = this.pDy, pR = this.pR, cap = pI.length;
    const ord = this.gOrd, cStart = this.cStart, cNbr = this.cNbr, nc = this.nCells;
    for (let i = 0; i < n; i++) rho[i] = M[i] * sig;
    for (let c = 0; c < nc; c++) {
      const s1 = cStart[c + 1];
      for (let a = cStart[c]; a < s1; a++) {
        const i = ord[a], xi = X[i], yi = Y[i];
        for (let k = 0; k < 5; k++) {
          let b0, b1;
          if (k === 0) { b0 = a + 1; b1 = s1; }
          else { const nb = cNbr[4 * c + k - 1]; if (nb < 0) continue; b0 = cStart[nb]; b1 = cStart[nb + 1]; }
          for (let b = b0; b < b1; b++) {
            const j = ord[b];
            const dx = xi - X[j], dy = yi - Y[j], r2 = dx * dx + dy * dy;
            if (r2 >= h2x4 || r2 === 0) continue;
            const r = Math.sqrt(r2), q = r * ih;
            let w;
            if (q < 1) w = sig * (1 - 1.5 * q * q + 0.75 * q * q * q); else { const t = 2 - q; w = sig * 0.25 * t * t * t; }
            rho[i] += M[j] * w; rho[j] += M[i] * w;
            if (np === cap) {              // pair list full (very dense shock): double it
              const grow = (A, T) => { const B = new T(cap * 2); B.set(A); return B; };
              pI = this.pI = grow(pI, Int32Array); pJ = this.pJ = grow(pJ, Int32Array);
              pDx = this.pDx = grow(pDx, Float64Array); pDy = this.pDy = grow(pDy, Float64Array); pR = this.pR = grow(pR, Float64Array);
              cap *= 2;
            }
            pI[np] = i; pJ[np] = j; pDx[np] = dx; pDy[np] = dy; pR[np] = r; np++;
          }
        }
      }
    }
    // 2) equation of state
    for (let i = 0; i < n; i++) {
      const cold = C0 * C0 * (rho[i] - R0[i]);
      P[i] = (cold > 0 ? cold : 0) + GAMMA_TH * rho[i] * U[i];
      cs[i] = Math.sqrt(C0 * C0 + GAMMA_TH * (GAMMA_TH + 1) * U[i]);
      DU[i] = 0;
    }
    // 3) pressure + artificial viscosity forces, heating
    let vsigMax = 1e-9;
    const eta2 = 0.01 * h * h;
    for (let p = 0; p < np; p++) {
      const i = pI[p], j = pJ[p], dx = pDx[p], dy = pDy[p], r = pR[p], q = r * ih;
      let dWdr;
      if (q < 2 / 3) dWdr = -fsig;                      // Thomas–Couchman: no pairing
      else if (q < 1) dWdr = fsig * (-3 * q + 2.25 * q * q);
      else { const t = 2 - q; dWdr = -0.75 * fsig * t * t; }
      const dw = dWdr / r;
      const dvx = VX[i] - VX[j], dvy = VY[i] - VY[j], vr = dvx * dx + dvy * dy;
      let pi = 0, vs = cs[i] + cs[j];
      if (vr < 0) {
        const mu = h * vr / (r * r + eta2);
        pi = (-AV_A * 0.5 * vs * mu + AV_B * mu * mu) / (0.5 * (rho[i] + rho[j]));
        vs -= 3 * vr / r;
      }
      if (vs > vsigMax) vsigMax = vs;
      const pij = P[i] / (rho[i] * rho[i]) + P[j] / (rho[j] * rho[j]) + pi;
      const fx = pij * dw * dx, fy = pij * dw * dy;
      AX[i] -= M[j] * fx; AY[i] -= M[j] * fy;
      AX[j] += M[i] * fx; AY[j] += M[i] * fy;
      const e = 0.5 * pij * dw * vr;
      DU[i] += M[j] * e; DU[j] += M[i] * e;
    }
    this.vsigMax = vsigMax;
  };

  // ---------------------------------------------------------------- phase 1
  Sim.prototype.forcesImpact = function () {
    const n = this.n, tE = this._tEval;       // tE: the time these positions belong to
    // Gravity changes slowly compared with the SPH time step, so the tree is only rebuilt
    // every gravDt (physical time, ~12 s) instead of every step. Holding it constant in
    // between would delay the restoring force and slowly pump energy into the debris, so
    // between rebuilds it is extrapolated linearly from the last two (error ~ gravDt^2).
    const age = tE - this._gravT;
    if (!(age >= 0 && age < this.o.gravDt) || this._gravN !== n) {
      this._havePrev = this.o.gravExtrap && this._gravN === n && age > 0;
      if (this._havePrev) { this.pgx.set(this.gax.subarray(0, n)); this.pgy.set(this.gay.subarray(0, n)); this._pT = this._gravT; }
      this.buildTree(); this.gravityTree();
      for (let i = 0; i < n; i++) { this.gax[i] = this.ax[i]; this.gay[i] = this.ay[i]; }
      this._gravT = tE; this._gravN = n;
    }
    const f = this._havePrev ? (tE - this._gravT) / (this._gravT - this._pT) : 0;
    if (f !== 0) for (let i = 0; i < n; i++) { this.ax[i] = this.gax[i] + f * (this.gax[i] - this.pgx[i]); this.ay[i] = this.gay[i] + f * (this.gay[i] - this.pgy[i]); }
    else for (let i = 0; i < n; i++) { this.ax[i] = this.gax[i]; this.ay[i] = this.gay[i]; }
    this.sphForces();
  };

  Sim.prototype.stepImpact = function (dtMax) {
    if (!this.h) this.sphSetup();
    if (!this._haveAcc) { this._tEval = this.t; this.forcesImpact(); this._haveAcc = true; }
    const dt = Math.min(dtMax, 0.3 * this.h / this.vsigMax);
    this._dt = dt;
    const n = this.n;
    for (let i = 0; i < n; i++) {
      this.vx[i] += 0.5 * dt * this.ax[i]; this.vy[i] += 0.5 * dt * this.ay[i];
      this.u[i] += 0.5 * dt * this.du[i]; if (this.u[i] < 0) this.u[i] = 0;
      this.x[i] += dt * this.vx[i]; this.y[i] += dt * this.vy[i];
    }
    this._tEval = this.t + dt; this.forcesImpact();
    for (let i = 0; i < n; i++) {
      this.vx[i] += 0.5 * dt * this.ax[i]; this.vy[i] += 0.5 * dt * this.ay[i];
      this.u[i] += 0.5 * dt * this.du[i]; if (this.u[i] < 0) this.u[i] = 0;
      if (this.o.tcool > 0) this.u[i] *= 1 - dt / this.o.tcool;
    }
    this.t += dt;
  };

  // ------------------------------------------------- phase 1 -> 2 transition
  // Like Canup's iterative procedure: estimate the planet's mass and radius,
  // then classify each particle by its orbit around the planet:
  //   unbound -> escapes;  bound with periapsis < R_planet -> planet;
  //   bound with periapsis >= R_planet -> disk.
  Sim.prototype.toAccretion = function () {
    const n = this.n;
    let M = 0, X = 0, Y = 0, VX = 0, VY = 0;
    for (let i = 0; i < n; i++) if (this.kind[i] === TGT_CORE || this.kind[i] === IMP_CORE) { const m = this.m[i]; M += m; X += m * this.x[i]; Y += m * this.y[i]; VX += m * this.vx[i]; VY += m * this.vy[i]; }
    X /= M; Y /= M; VX /= M; VY /= M;
    // radius: area occupied by dense (condensed) material near the centre
    let A = 0;
    for (let i = 0; i < n; i++) if (this.rho[i] > 0.6 * this.rho0[i] && Math.hypot(this.x[i] - X, this.y[i] - Y) < 2.5) A += this.m[i] / this.rho[i];
    const Rp = Math.sqrt(A / Math.PI);
    const cls = new Uint8Array(n);   // 0 planet, 1 disk, 2 escape
    let Mp = 0; for (let i = 0; i < n; i++) Mp += this.m[i];
    for (let it = 0; it < 4; it++) {
      let Mnew = 0, Xn = 0, Yn = 0, VXn = 0, VYn = 0;
      for (let i = 0; i < n; i++) {
        const dx = this.x[i] - X, dy = this.y[i] - Y, dvx = this.vx[i] - VX, dvy = this.vy[i] - VY;
        const r = Math.hypot(dx, dy), v2 = dvx * dvx + dvy * dvy;
        const E = 0.5 * v2 - Mp / r;
        if (r < Rp) cls[i] = 0;
        else if (E >= 0) cls[i] = 2;
        else {
          const L = dx * dvy - dy * dvx, a = -Mp / (2 * E), e = Math.sqrt(Math.max(0, 1 + 2 * E * L * L / (Mp * Mp)));
          cls[i] = a * (1 - e) < Rp ? 0 : 1;
        }
        if (cls[i] === 0) { const m = this.m[i]; Mnew += m; Xn += m * this.x[i]; Yn += m * this.y[i]; VXn += m * this.vx[i]; VYn += m * this.vy[i]; }
      }
      Mp = Mnew; X = Xn / Mnew; Y = Yn / Mnew; VX = VXn / Mnew; VY = VYn / Mnew;
    }
    let Ft = 0, mm = 0, mf = 0, Up = 0;
    for (let i = 0; i < n; i++) if (cls[i] === 0) {
      Ft += this.m[i] * this.ftar[i]; Up += this.m[i] * this.u[i];
      if (this.kind[i] === TGT_MANTLE || this.kind[i] === IMP_MANTLE) { mm += this.m[i]; mf += this.m[i] * this.ftar[i]; }
    }
    const old = {}; for (const k of ['x', 'y', 'vx', 'vy', 'm', 'r', 'u', 'ftar', 'kind']) old[k] = this[k].slice(0, n);
    this.n = 0;
    this.add(X, Y, VX, VY, Mp, Rp, 255, Ft / Mp);
    this.u[0] = Up / Mp;                       // the merged planet keeps its mean heat
    this.planet = { R: Rp, roche: 2.9 * Rp, mantleFt: mf / mm, M: Mp };
    let nd = 0, ne = 0, ftd = 0, md = 0;
    for (let i = 0; i < n; i++) {
      if (cls[i] === 2) { ne++; continue; }
      if (cls[i] !== 1) continue;
      this.add(old.x[i], old.y[i], old.vx[i], old.vy[i], old.m[i], old.r[i], old.kind[i], old.ftar[i]);
      this.u[this.n - 1] = old.u[i];
      nd++;
      if (old.kind[i] === TGT_MANTLE || old.kind[i] === IMP_MANTLE) { md += old.m[i]; ftd += old.m[i] * old.ftar[i]; }
    }
    this.phase = 2;
    this.diskInfo = { n: nd, M: this.diskMass(), mantleFt: md ? ftd / md : NaN, escaped: ne };
    this._haveAcc = false;
    this.log.push(`t=${this.hours().toFixed(1)} h: planet ${Mp.toFixed(3)} M⊕ (R=${Rp.toFixed(2)}), disk ${nd} particles = ${(this.diskInfo.M / M_LUNAR).toFixed(2)} lunar masses, ${ne} escaped`);
  };

  // ---------------------------------------------------------------- phase 2
  Sim.prototype.forcesAccretion = function () {
    const n = this.n, eps2 = this.eps * this.eps;
    const X = this.x, Y = this.y, M = this.m, AX = this.ax, AY = this.ay;
    for (let i = 0; i < n; i++) { AX[i] = 0; AY[i] = 0; }
    for (let i = 0; i < n; i++) {
      const xi = X[i], yi = Y[i], mi = M[i];
      for (let j = i + 1; j < n; j++) {
        const dx = X[j] - xi, dy = Y[j] - yi, r2 = dx * dx + dy * dy + eps2;
        const inv = 1 / (r2 * Math.sqrt(r2));
        AX[i] += dx * inv * M[j]; AY[i] += dy * inv * M[j];
        AX[j] -= dx * inv * mi; AY[j] -= dy * inv * mi;
      }
    }
    for (let i = 1; i < n; i++) for (let j = i + 1; j < n; j++) contact(this, i, j);
  };

  Sim.prototype.mergeInto = function (a, b) {        // b is absorbed into a
    const ma = this.m[a], mb = this.m[b], m = ma + mb;
    if (a !== 0) {
      this.x[a] = (ma * this.x[a] + mb * this.x[b]) / m; this.y[a] = (ma * this.y[a] + mb * this.y[b]) / m;
    }
    this.vx[a] = (ma * this.vx[a] + mb * this.vx[b]) / m; this.vy[a] = (ma * this.vy[a] + mb * this.vy[b]) / m;
    this.ftar[a] = (ma * this.ftar[a] + mb * this.ftar[b]) / m;
    this.u[a] = (ma * this.u[a] + mb * this.u[b]) / m;
    this.m[a] = m;
    if (a !== 0) { this.r[a] = this.r0 * Math.sqrt(m / this.m0); this.kind[a] = MOONLET; }
    this.remove(b);
  };

  Sim.prototype.accretionEvents = function () {
    const P = 0, Rp = this.r[P], roche = this.planet.roche;
    // 1) debris hitting the planet, or flying far away
    for (let i = this.n - 1; i >= 1; i--) {
      const d = Math.hypot(this.x[i] - this.x[P], this.y[i] - this.y[P]);
      if (d < Rp + 0.5 * this.r[i]) this.mergeInto(P, i);
      else if (d > 200) this.remove(i);
    }
    // 2) gravitationally bound pairs beyond the Roche limit stick together
    const Mp = this.m[P];
    for (let i = 1; i < this.n; i++) {
      for (let j = i + 1; j < this.n; j++) {
        const dx = this.x[j] - this.x[i], dy = this.y[j] - this.y[i], d = Math.hypot(dx, dy);
        const mi = this.m[i], mj = this.m[j];
        const mx = (mi * this.x[i] + mj * this.x[j]) / (mi + mj) - this.x[P], my = (mi * this.y[i] + mj * this.y[j]) / (mi + mj) - this.y[P];
        const a = Math.hypot(mx, my);
        if (a < roche) continue;
        const rH = a * Math.cbrt((mi + mj) / (3 * Mp));
        const touch = this.r[i] + this.r[j];
        if (d > Math.max(1.05 * touch, this.o.hill * rH)) continue;
        // merge if the pair is slow enough to be captured: relative speed below
        // their mutual escape speed (bodies touching) or a few times the Hill
        // speed (bodies inside each other's Hill sphere, i.e. a gravitationally
        // collapsing clump)
        const dv = Math.hypot(this.vx[j] - this.vx[i], this.vy[j] - this.vy[i]);
        const vEsc = Math.sqrt(2 * (mi + mj) / Math.max(d, touch));
        const vHill = 6 * rH * Math.sqrt(Mp / (a * a * a));
        if (dv < Math.max(vEsc, vHill)) {
          if (mi >= mj) this.mergeInto(i, j); else { this.mergeInto(j, i); }
          j = i; // restart inner scan for i (array changed)
          if (i >= this.n) break;
        }
      }
    }
  };

  // The real disk is a dense, partly vaporised melt: collisions and gas
  // pressure damp eccentric orbits (circularisation, angular momentum kept),
  // and material inside the Roche limit, where tides stop it from clumping,
  // spreads viscously and drains onto the planet. Modelled with two timescales.
  Sim.prototype.diskDamping = function (dt) {
    const P = 0, tc = this.o.tCirc, tv = this.o.tVisc, roche = this.planet.roche;
    for (let i = 1; i < this.n; i++) {
      const dx = this.x[i] - this.x[P], dy = this.y[i] - this.y[P];
      const r = Math.hypot(dx, dy), ex = dx / r, ey = dy / r;
      let vx = this.vx[i] - this.vx[P], vy = this.vy[i] - this.vy[P];
      let vr = vx * ex + vy * ey, vt = -vx * ey + vy * ex;
      vr *= 1 - dt / tc;                                   // circularise
      if (r < roche) vt *= 1 - dt / tv;                    // inner disk drains inward
      this.vx[i] = this.vx[P] + vr * ex - vt * ey;
      this.vy[i] = this.vy[P] + vr * ey + vt * ex;
    }
  };

  Sim.prototype.stepAccretion = function (dt) {
    this._dt = dt;
    if (!this._haveAcc) { this.forcesAccretion(); this._haveAcc = true; }
    const n = this.n;
    for (let i = 0; i < n; i++) {
      this.vx[i] += 0.5 * dt * this.ax[i]; this.vy[i] += 0.5 * dt * this.ay[i];
      this.x[i] += dt * this.vx[i]; this.y[i] += dt * this.vy[i];
    }
    this.forcesAccretion();
    for (let i = 0; i < n; i++) { this.vx[i] += 0.5 * dt * this.ax[i]; this.vy[i] += 0.5 * dt * this.ay[i]; }
    this.diskDamping(dt);
    this.t += dt;
    const before = this.n;
    this.accretionEvents();
    if (this.n !== before) this._haveAcc = false;
  };

  // ---------------------------------------------------------------- driver
  Sim.prototype.step = function () {
    if (this.phase === 1) {
      this.stepImpact(this.o.dt);
      for (let i = this.n - 1; i >= 0; i--) if (Math.abs(this.x[i]) > 80 || Math.abs(this.y[i]) > 80) { this.remove(i); this._haveAcc = false; }
      if (this.t >= this.o.tSwitch) this.toAccretion();
    } else if (this.phase === 2) {
      this.stepAccretion(this.o.dt2);
    }
  };

  // ---------------------------------------------------------------- stats
  Sim.prototype.temperature = function (i) { return T_BASE + this.u[i] * T_PER_U; };
  Sim.prototype.tempStats = function () {   // hottest particle and mass-weighted mean, in K
    let mx = 0, sm = 0, M = 0;
    for (let i = 0; i < this.n; i++) { if (this.u[i] > mx) mx = this.u[i]; sm += this.m[i] * this.u[i]; M += this.m[i]; }
    return { max: T_BASE + mx * T_PER_U, mean: T_BASE + (M ? sm / M : 0) * T_PER_U };
  };
  Sim.prototype.hours = function () { return this.t * T_UNIT_S / 3600; };
  Sim.prototype.diskMass = function () { let s = 0; for (let i = 1; i < this.n; i++) s += this.m[i]; return this.phase === 2 ? s : NaN; };
  Sim.prototype.largestMoon = function () {
    if (this.phase !== 2) return null;
    let k = -1, mm = 0;
    for (let i = 1; i < this.n; i++) if (this.m[i] > mm) { mm = this.m[i]; k = i; }
    if (k < 0) return null;
    const dx = this.x[k] - this.x[0], dy = this.y[k] - this.y[0];
    return { i: k, m: mm, lunar: mm / M_LUNAR, dist: Math.hypot(dx, dy) / this.r[0], ftar: this.ftar[k] };
  };

  Sim.consts = { T_BASE, T_PER_U, T_UNIT_S, V_UNIT_KMS, M_LUNAR, TGT_MANTLE, TGT_CORE, IMP_MANTLE, IMP_CORE, MOONLET };
  root.MoonSim = Sim;
  if (typeof module !== 'undefined') module.exports = Sim;
})(typeof window !== 'undefined' ? window : globalThis);